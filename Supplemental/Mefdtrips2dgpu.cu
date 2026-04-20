/*2D elastic time-domain FD time-reverse imaging (PS energy norm imaging condition) with GPU*/

/*
  Authors: Can Oren, Robin M. Weiss, and Jeffrey Shragge

  This code is a GPU-enabled version of the efdtri2d module from the Madagascar
  software package (see: http://www.reproducibility.org).  It implements a 2D
  Finite-Difference Time Domain solver for the elastic time-reverse wave equation 
  with 2nd- and 8th- order temporal and spatial accuracy, respectively.
*/

/*
  Copyright (C) 2019 Colorado School of Mines
  
  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.
  
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/
#include <cuda.h>
#include <cuda_runtime_api.h>

extern "C" {
#include <rsf.h>
}

#include "fdutil.c"
#include "efdtri2d_kernels.cu"

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define NOP 4 /* derivative operator half-size */


// checks the current GPU device for an error flag and prints to stderr
static void sf_check_gpu_error (const char *msg) {
    cudaError_t err = cudaGetLastError ();
    if (cudaSuccess != err)
        sf_error ("Cuda error: %s: %s", msg, cudaGetErrorString (err));
}

// entry point
int main(int argc, char* argv[]) {
	
    bool verb,fsrf,snap,dabc,opot,taper;
    int  jsnap,ntsnap,jdata;

    /* I/O files */
    sf_file Frec=NULL;  /* receivers */
    sf_file Fccc=NULL;  /* velocity */
    sf_file Fden=NULL;  /* density */
    sf_file Fdatpz=NULL; /* vertical P-wave data */
    sf_file Fdatpx=NULL; /* horizontal P-wave data */
	sf_file Fdatsz=NULL; /* vertical S-wave data */
    sf_file Fdatsx=NULL; /* horizontal S-wave data */
    sf_file Fimg=NULL;  /* image */
    sf_file Feimg=NULL; /* extended image */
    sf_file Fwfl=NULL;  /* wavefield */
    sf_file Fcip=NULL;

    /* cube axes */
    sf_axis at,ax,az;
    sf_axis ar,ac;

    int     nt,nz,nx,nr,nc,nb;
    int     it,iz,ix;
    float   dt,dz,dx,idz,idx;
	float   tmin,tmax;
	int     ntap,nttap,idxt2;

    /* FDM structure */
    fdm2d    fdm=NULL;
    abcone2d abcs=NULL;
    sponge   spo=NULL;

    /* I/O arrays */
    pt2d   *rr=NULL;           /* receivers */

    /*------------------------------------------------------------*/
    /* orthorombic footprint - 4 coefficients */
    /* c11 c13 
       .   c33 
               c55 */
	float *h_c11, *h_c33, *h_c55, *h_c13;
	float *d_c11, *d_c33, *d_c55, *d_c13;
    float **vs=NULL;

	// density
	float *h_ro, *d_ro, *d_ro2;

    /*------------------------------------------------------------*/
    /* displacement: um = U @ t-1; uo = U @ t; up = U @ t+1 */

	float *d_umzp, *d_uozp, *d_upzp, *d_uazp, *d_utzp;
	float *d_umxp, *d_uoxp, *d_upxp, *d_uaxp, *d_utxp;
    float *d_uvzp, *d_uvxp;
	float *d_umzs, *d_uozs, *d_upzs, *d_uazs, *d_utzs;
    float *d_umxs, *d_uoxs, *d_upxs, *d_uaxs, *d_utxs;
    float *d_uvzs, *d_uvxs;

    float *d_potp, *d_pots;
	
	// used for writing wavefield to output file
	float *h_uoz, *h_uox; 
	float **uoz, **uox;
    float ****uez;
	
    /* stress/strain tensor */ 
	float *d_tzzp, *d_tzxp, *d_txxp;
    float *d_dzzp, *d_dzxp, *d_dxxp;
	float *d_tzzs, *d_tzxs, *d_txxs;
    float *d_dzzs, *d_dzxs, *d_dxxs;

    
    /*------------------------------------------------------------*/
    /* linear interpolation weights/indices */
    lint2d cr;

    /* Gaussian bell */
    int nbell;

    /* imaging condition type */
    int  ictype;
    int eictype;

    /* wavefield cut params */
    sf_axis   acz=NULL,acx=NULL;
    sf_axis   aiz=NULL,aix=NULL;
    sf_axis   ahz=NULL,ahx=NULL,alz=NULL,alx=NULL;
    sf_axis   an=NULL;
    //sf_axis   aimgz=NULL,aimgx=NULL;

    int       nqz,nqx;
    int       nhz,nhx;
	int		  nlz,nlx;
    float     oqz,oqx;
    float     dqz,dqx;
    float     **uc=NULL;

    /* extended image parameters */
    int     cipz,cipx;
    float   lhz,lhx;

    /*------------------------------------------------------------*/
    /* init RSF */
    sf_init(argc,argv);


    /*------------------------------------------------------------*/
    /* execution flags */
    if(! sf_getbool("verb",&verb))  verb=false; /* verbosity flag */
    if(! sf_getbool("snap",&snap))  snap=false; /* wavefield snapshots flag */
    if(! sf_getbool("free",&fsrf))  fsrf=false; /* free surface flag */
    if(! sf_getbool("dabc",&dabc))  dabc=false; /* absorbing BC */
    if(! sf_getbool("opot",&opot))  opot=false; /* wavefield potentials flag */
    /* imaging condition type: 
		1 = Xcorr
		2 = Xcorr-P_ver 
		3 = Xcorr-S_hor
		4 = Energy Norm (Laplacian)
		5 = Kinetic Energy
		6 = Potential Energy
		7 = Energy Norm (Hamiltonian)
		8 = Energy Ratio (Laplacian/Hamiltonian)
	*/
    if(! sf_getint("ictype",&ictype)) ictype=4; 
    if(! sf_getint("eictype",&eictype)) eictype=2; /* extended imaging condition type: 1=crosscorrelation; 2=energy norm */
	if(! sf_getbool("taper",&taper)) taper=false; /* taper flag */
    /*------------------------------------------------------------*/


    /*------------------------------------------------------------*/
    /* I/O files */
    Fccc   = sf_input ("in");    /* stiffness */
    Fden   = sf_input ("den");   /* density   */
    Frec   = sf_input ("rec");   /* receivers */
    Fdatpz = sf_input ("datpz"); /* vertical P-wave data */
    Fdatpx = sf_input ("datpx"); /* horizontal P-wave data */
	Fdatsz = sf_input ("datsz"); /* vertical S-wave data */
    Fdatsx = sf_input ("datsx"); /* horizontal S-wave data */
    Fimg   = sf_output("out");   /* image */
    Feimg  = sf_output("eimg");  /* extended image */
    Fcip   = sf_input("cip");    /* common-image-point coordinates */
    if(snap)
	Fwfl = sf_output("bwfl"); /* wavefield */
    
    /*------------------------------------------------------------*/


    /*------------------------------------------------------------*/
    /* init GPU */
	int gpu;
	if (! sf_getint("gpu", &gpu)) gpu = 0;	/* ID of the GPU to be used */
	sf_warning("using GPU #%d", gpu);
	cudaSetDevice(gpu);
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	cudaStream_t stream[6];
	for (int i = 0; i < 6; ++i) {
		cudaStreamCreate(&stream[i]);
	}
	

    /*------------------------------------------------------------*/
    /* axes */
    at = sf_iaxa(Fdatpz,2); sf_setlabel(at,"t"); if(verb) sf_raxa(at); /* time  */
    az = sf_iaxa(Fccc,1);   sf_setlabel(az,"z"); if(verb) sf_raxa(az); /* depth */
    ax = sf_iaxa(Fccc,2);   sf_setlabel(ax,"x"); if(verb) sf_raxa(ax); /* space */
    ar = sf_iaxa(Frec,2);   sf_setlabel(ar,"r"); if(verb) sf_raxa(ar); /* receivers */

    nt = sf_n(at); dt = sf_d(at);
    nz = sf_n(az); dz = sf_d(az);
    nx = sf_n(ax); dx = sf_d(ax);

    nr = sf_n(ar);

	//niz=sf_n(az);
    //nix=sf_n(ax);

    //oiz=sf_o(az);
    //oix=sf_o(ax);

    //diz=sf_d(az);
    //dix=sf_d(ax);
    /*------------------------------------------------------------*/


    /*------------------------------------------------------------*/
    /* other execution parameters */
    if(! sf_getint("nbell",&nbell)) nbell=5;  /* bell size */
    if(verb) sf_warning("nbell=%d",nbell);
    if(! sf_getint("jdata",&jdata)) jdata=1;	/* extract receiver data every jdata time steps */
    if(snap) {  
		if(! sf_getint("jsnap",&jsnap)) jsnap=nt;	/* save wavefield every jsnap time steps */
    }
	if(! sf_getfloat("tmin",&tmin)) tmin=0; /* end time for imaging condition */
    if(! sf_getfloat("tmax",&tmax)) tmax=nt*dt; /* start time for imaging condition */
	if(! sf_getint("ntap",&ntap)) ntap=40; /* taper length */
	//if(! sf_getint("ncip",&ncip)) ncip=1; /* number of common image points next to the central coordinate */
	//ncip2 = (2*ncip+1)*(2*ncip+1);
	//if(! sf_getint("nlz",&nlz)) nlz=3;
	//if(! sf_getint("nlx",&nlx)) nlx=3;
    /*------------------------------------------------------------*/


    /*------------------------------------------------------------*/
    /* expand domain for FD operators and ABC */
    if( !sf_getint("nb",&nb) || nb<NOP) nb=NOP;

    fdm=fdutil_init(verb,fsrf,az,ax,nb,1);

	if (nbell * 2 + 1 > 32){
		sf_error("nbell must be <= 15\n"); 
	}

	float *h_bell;
	float *d_bell;
	h_bell = (float*)malloc((2*nbell+1)*(2*nbell+1)*sizeof(float));
	float s = 0.5*nbell;
	for (ix=-nbell;ix<=nbell;ix++) {
		for (iz=-nbell;iz<=nbell;iz++) {
	    	h_bell[(iz + nbell) * (2*nbell+1) + (ix + nbell)] = exp(-(iz*iz+ix*ix)/s);
		}
	}
	cudaMalloc((void**)&d_bell, (2*nbell+1)*(2*nbell+1)*sizeof(float));
	cudaMemcpy(d_bell, h_bell, (2*nbell+1)*(2*nbell+1)*sizeof(float), cudaMemcpyHostToDevice);

    sf_setn(az,fdm->nzpad); sf_seto(az,fdm->ozpad); if(verb) sf_raxa(az);
    sf_setn(ax,fdm->nxpad); sf_seto(ax,fdm->oxpad); if(verb) sf_raxa(ax);
    /*------------------------------------------------------------*/

    /*------------------------------------------------------------*/
    /* NULL AXIS */
    an=sf_maxa(1,0,1);
    /*------------------------------------------------------------*/

    /*------------------------------------------------------------*/
    /* 2D vector components */
    nc=2;
    ac=sf_maxa(nc,0,1);
    /*------------------------------------------------------------*/


    /* setup output wavefield header and arrays*/
    if(snap) {
	
		uoz=sf_floatalloc2(fdm->nzpad,fdm->nxpad);
		uox=sf_floatalloc2(fdm->nzpad,fdm->nxpad);

		h_uoz = (float*)malloc(fdm->nzpad * fdm->nxpad * sizeof(float));
		h_uox = (float*)malloc(fdm->nzpad * fdm->nxpad * sizeof(float));
	
		nqz=sf_n(az);
		nqx=sf_n(ax);
	    
		oqz=sf_o(az);
		oqx=sf_o(ax);
	
		dqz=sf_d(az);
		dqx=sf_d(ax);
	
		acz = sf_maxa(nqz,oqz,dqz); if(verb)sf_raxa(acz);
		acx = sf_maxa(nqx,oqx,dqx); if(verb)sf_raxa(acx);
	
		uc=sf_floatalloc2(sf_n(acz),sf_n(acx));
	
		ntsnap=0;
		for(it=0; it<nt; it++) {
		    if(it%jsnap==0) ntsnap++;
		}
		sf_setn(at,  ntsnap);
		sf_setd(at,dt*jsnap);
		if(verb) sf_raxa(at);
	
		sf_oaxa(Fwfl,acz,1);
		sf_oaxa(Fwfl,acx,2);
		sf_oaxa(Fwfl,ac, 3);
		sf_oaxa(Fwfl,at, 4);
    }

    /* set axes for image output */
    const char* lbl_x = "distance";
    const char* unt_x = "km";
    const char* lbl_z = "depth";
    const char* unt_z = "km";
		
    aiz = sf_maxa(nqz,oqz,dqz); sf_setlabel(aiz,lbl_z); sf_setunit(aiz,unt_z); if(verb)sf_raxa(aiz);
    aix = sf_maxa(nqx,oqx,dqx); sf_setlabel(aix,lbl_x); sf_setunit(aix,unt_x); if(verb)sf_raxa(aix);
	sf_oaxa(Fimg,aiz,1);
	sf_oaxa(Fimg,aix,2);
	sf_oaxa(Fimg,an ,3);

    /* set axes for extended image output */
    //if(! sf_getint("cipz",&cipz)) cipz=175; /* common-image-point-z */
    //if(! sf_getint("cipx",&cipx)) cipx=300; /* common-image-point-x */
    float cdum;
    sf_floatread(&cdum,1,Fcip);
    cipx = (int) cdum;
    sf_floatread(&cdum,1,Fcip);
    cipz = (int) cdum;
    if(verb) sf_warning("cipz: %i cipx: %i",cipz,cipx);
    if(! sf_getfloat("lhz",&lhz)) lhz=0.15; /* vertical space-lag   */
    if(! sf_getfloat("lhx",&lhx)) lhx=0.15; /* horizontal space-lag */    

        nhz=lhz/dz;
        ahz=sf_maxa(2*nhz+1,-nhz*dz,dz); sf_setlabel(ahz,"hz"); sf_setunit(ahz,"km");
        if(verb) sf_raxa(ahz);

        nhx=lhx/dx;
        ahx=sf_maxa(2*nhx+1,-nhx*dx,dx); sf_setlabel(ahx,"hx"); sf_setunit(ahx,"km");
        if(verb) sf_raxa(ahx);

		//ahc=sf_maxa(ncip2,0,1);
		nlz=2*nhz+1;
		alz=sf_maxa(nlz,0,dz); sf_setlabel(alz,"lz"); sf_setunit(alz,"km");
		if(verb) sf_raxa(alz);

		nlx=2*nhx+1;
		alx=sf_maxa(nlx,0,dz); sf_setlabel(alx,"lx"); sf_setunit(alx,"km");
        if(verb) sf_raxa(alx);

		sf_oaxa(Feimg,alz,1);
		sf_oaxa(Feimg,alx,2);
        sf_oaxa(Feimg,ahz,3);
        sf_oaxa(Feimg,ahx,4);

        //uez=sf_floatalloc3(sf_n(ahc),sf_n(ahz),sf_n(ahx));
		uez=sf_floatalloc4(sf_n(alz),sf_n(alx),sf_n(ahz),sf_n(ahx));

    /*------------------------------------------------------------*/
    /* read in data and copy to GPU */
    float *d_dd_pz; 
    float *d_dd_px;
    float *h_dd_pz;
    float *h_dd_px;
    h_dd_pz = (float*)malloc(nr*nt*sizeof(float));
    h_dd_px = (float*)malloc(nr*nt*sizeof(float));
    sf_floatread(h_dd_pz,nt*nr,Fdatpz);
    sf_floatread(h_dd_px,nt*nr,Fdatpx);
    cudaMalloc((void**)&d_dd_pz, nr*nt*sizeof(float));
    cudaMalloc((void**)&d_dd_px, nr*nt*sizeof(float));
    cudaMemcpy(d_dd_pz, h_dd_pz, nr*nt*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dd_px, h_dd_px, nr*nt*sizeof(float), cudaMemcpyHostToDevice);
    /*------------------------------------------------------------*/

	/*------------------------------------------------------------*/
    /* read in data and copy to GPU */
    float *d_dd_sz;
    float *d_dd_sx;
    float *h_dd_sz;
    float *h_dd_sx;
    h_dd_sz = (float*)malloc(nr*nt*sizeof(float));
    h_dd_sx = (float*)malloc(nr*nt*sizeof(float));
    sf_floatread(h_dd_sz,nt*nr,Fdatsz);
    sf_floatread(h_dd_sx,nt*nr,Fdatsx);
    cudaMalloc((void**)&d_dd_sz, nr*nt*sizeof(float));
    cudaMalloc((void**)&d_dd_sx, nr*nt*sizeof(float));
    cudaMemcpy(d_dd_sz, h_dd_sz, nr*nt*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dd_sx, h_dd_sx, nr*nt*sizeof(float), cudaMemcpyHostToDevice);
    /*------------------------------------------------------------*/


    /*------------------------------------------------------------*/
    /* image arrays */
    float *h_img = NULL;
    h_img = (float*)malloc(fdm->nzpad*fdm->nxpad*sizeof(float));
    memset(h_img,0,fdm->nzpad*fdm->nxpad*sizeof(float));

    float *d_img;
    cudaMalloc((void**) &d_img,fdm->nzpad*fdm->nxpad*sizeof(float));
    cudaMemset(d_img,0,fdm->nzpad*fdm->nxpad*sizeof(float));
    /*------------------------------------------------------------*/


    /*------------------------------------------------------------*/
    /* extended image arrays */
    float *h_eimg = NULL;
    h_eimg = (float*)malloc(sf_n(alz)*sf_n(alx)*sf_n(ahz)*sf_n(ahx)*sizeof(float));
    memset(h_eimg,0,sf_n(alz)*sf_n(alx)*sf_n(ahz)*sf_n(ahx)*sizeof(float));

    float *d_eimg;
    cudaMalloc((void**) &d_eimg,sf_n(alz)*sf_n(alx)*sf_n(ahz)*sf_n(ahx)*sizeof(float));
    cudaMemset(d_eimg,0,sf_n(alz)*sf_n(alx)*sf_n(ahz)*sf_n(ahx)*sizeof(float));
    /*------------------------------------------------------------*/


    /*------------------------------------------------------------*/
    /* setup receiver coordinates */
    rr = (pt2d*) sf_alloc(nr,sizeof(*rr));
    pt2dread1(Frec,rr,nr,2); /* read (x,z) coordinates */
	
	
    /* calculate 2d linear interpolation coefficients for receiver locations */
    cr = lint2d_make(nr,rr,fdm);

    float *d_Rw00, *d_Rw01, *d_Rw10, *d_Rw11;
    cudaMalloc((void**)&d_Rw00, nr * sizeof(float));
	cudaMalloc((void**)&d_Rw01, nr * sizeof(float));
	cudaMalloc((void**)&d_Rw10, nr * sizeof(float));
	cudaMalloc((void**)&d_Rw11, nr * sizeof(float));
	cudaMemcpy(d_Rw00, cr->w00, nr * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Rw01, cr->w01, nr * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Rw10, cr->w10, nr * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Rw11, cr->w11, nr * sizeof(float), cudaMemcpyHostToDevice);
	
	// z and x coordinates of each receiver
	int *d_Rjz, *d_Rjx;
	cudaMalloc((void**)&d_Rjz, nr * sizeof(int));
	cudaMalloc((void**)&d_Rjx, nr * sizeof(int));
	cudaMemcpy(d_Rjz, cr->jz, nr * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Rjx, cr->jx, nr * sizeof(int), cudaMemcpyHostToDevice);

	/**************************************************************/
    /* Initialize partial image taper and copy to GPU */
    sf_complex *tap_h=NULL; /* taper on CPU */
    //cuComplex  *tap_d; /* taper on GPU */
    //nttap = (tmax - tmin)/dt;
    int idxt = 0;
    for (it=nt-1; it>=0; it--) {
        if(it*dt < tmax && it*dt >= tmin) {
        idxt += 1;
        }
    }
    nttap = idxt;
    tap_h = sf_complexalloc(nttap);
    sf_warning("tmin= %f, tmax= %f, dt= %f, nttap= %d \n", tmin, tmax, dt, nttap);
    int j1;

    // if taper='n'
    for (int ix=0; ix < nttap; ix++) {tap_h[ix]= sf_cmplx(1.f,0.f);}

    if(taper) {

    for (int ix=ntap; ix<nttap; ix++) {tap_h[ix]= sf_cmplx(1.f,0.f);}

    for (int ix=0; ix < ntap; ix++) {
        j1 = abs(ntap-ix-1.f);
        tap_h[ix] = sf_cmplx( cos ( SF_PI/2.f*(float)j1/((float)ntap-1.f) ), 0.f);
    }

    for (int ix=0; ix < ntap; ix++) {
        j1 = abs(ntap-ix-1.f);
        tap_h[nttap-ix-1] = sf_cmplx( cos (SF_PI/2.f*(float)j1/((float)ntap-1.f) ),0.f);
    }

    tap_h[    0]=sf_cmplx(0.f,0.f);
    tap_h[nttap-1]=sf_cmplx(0.f,0.f);

	//cudaMalloc((void **)&tap_d, nxx*sizeof(cuComplex) );
    //cudaMemcpy( tap_d, tap_h, nxx*sizeof(cuComplex), cudaMemcpyHostToDevice );
    //sf_warning("FINISHED TAPER");

    }

    //for (int ix=0; ix < nttap; ix++) {
    //    sf_warning("taper= %f \n", crealf(tap_h[ix]));
    //}	

    /*------------------------------------------------------------*/
    /* setup FD coefficients */
    idz = 1/dz;
    idx = 1/dx;
    /*------------------------------------------------------------*/ 
	
	
	/*------------------------------------------------------------*/ 
	/* Read density and stiffness model data and transfer to GPU */
	
	float *tt1 = (float*)malloc(nz * nx * sizeof(float)); 
	h_ro=(float*)malloc(fdm->nzpad * fdm->nxpad * sizeof(float));
	h_c11=(float*)malloc(fdm->nzpad * fdm->nxpad * sizeof(float));
	h_c33=(float*)malloc(fdm->nzpad * fdm->nxpad * sizeof(float));
	h_c55=(float*)malloc(fdm->nzpad * fdm->nxpad * sizeof(float));
	h_c13=(float*)malloc(fdm->nzpad * fdm->nxpad * sizeof(float));

    /* input density */
    sf_floatread(tt1,nz*nx,Fden);     expand_cpu(tt1,h_ro , fdm->nb, nx, fdm->nxpad, nz, fdm->nzpad);

    /* input stiffness */
    sf_floatread(tt1,nz*nx,Fccc );    expand_cpu(tt1,h_c11, fdm->nb, nx, fdm->nxpad, nz, fdm->nzpad);
    sf_floatread(tt1,nz*nx,Fccc );    expand_cpu(tt1,h_c33, fdm->nb, nx, fdm->nxpad, nz, fdm->nzpad);
    sf_floatread(tt1,nz*nx,Fccc );    expand_cpu(tt1,h_c55, fdm->nb, nx, fdm->nxpad, nz, fdm->nzpad);
    sf_floatread(tt1,nz*nx,Fccc );    expand_cpu(tt1,h_c13, fdm->nb, nx, fdm->nxpad, nz, fdm->nzpad);
	
	cudaMalloc((void **)&d_ro, fdm->nzpad * fdm->nxpad * sizeof(float));
    cudaMalloc((void **)&d_ro2, fdm->nzpad * fdm->nxpad * sizeof(float));
	cudaMalloc((void **)&d_c11, fdm->nzpad * fdm->nxpad * sizeof(float));
	cudaMalloc((void **)&d_c33, fdm->nzpad * fdm->nxpad * sizeof(float));
	cudaMalloc((void **)&d_c55, fdm->nzpad * fdm->nxpad * sizeof(float));
	cudaMalloc((void **)&d_c13, fdm->nzpad * fdm->nxpad * sizeof(float));
	
	cudaMemcpy(d_ro, h_ro, fdm->nzpad * fdm->nxpad * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c11, h_c11, fdm->nzpad * fdm->nxpad * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c33, h_c33, fdm->nzpad * fdm->nxpad * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c55, h_c55, fdm->nzpad * fdm->nxpad * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_c13, h_c13, fdm->nzpad * fdm->nxpad * sizeof(float), cudaMemcpyHostToDevice);


    /*------------------------------------------------------------*/
	/* boundary condition setup */
   	float *d_bzl_s, *d_bzh_s;
	float *d_bxl_s, *d_bxh_s;

	float *d_spo;
    if(dabc) {
		/* one-way abc setup   */
		vs = sf_floatalloc2(fdm->nzpad,fdm->nxpad); 
		for (ix=0; ix<fdm->nxpad; ix++) {
		    for(iz=0; iz<fdm->nzpad; iz++) {
				vs[ix][iz] = sqrt(h_c55[iz * fdm->nxpad + ix]/h_ro[iz * fdm->nxpad + ix] );
		    }
		}
		abcs = abcone2d_make(NOP,dt,vs,fsrf,fdm);
		free(*vs); free(vs);

		cudaMalloc((void**)&d_bzl_s, fdm->nxpad * sizeof(float));
		cudaMalloc((void**)&d_bzh_s, fdm->nxpad * sizeof(float));
		cudaMalloc((void**)&d_bxl_s, fdm->nzpad * sizeof(float));
		cudaMalloc((void**)&d_bxh_s, fdm->nzpad * sizeof(float));
		
		cudaMemcpy(d_bzl_s, abcs->bzl, fdm->nxpad * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_bzh_s, abcs->bzh, fdm->nxpad * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_bxl_s, abcs->bxl, fdm->nzpad * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_bxh_s, abcs->bxh, fdm->nzpad * sizeof(float), cudaMemcpyHostToDevice);
		
		/* sponge abc setup */
		spo = sponge_make(fdm->nb);
		
		// d_spo contains all of the sponge coefficients
		cudaMalloc((void**)&d_spo, fdm->nb * sizeof(float));
		cudaMemcpy(d_spo, spo->w, fdm->nb * sizeof(float), cudaMemcpyHostToDevice);
    }


    /*------------------------------------------------------------*/
    /* allocate wavefield arrays */
	cudaMalloc((void **)&d_umzp, fdm->nzpad * fdm->nxpad * sizeof(float));
	cudaMalloc((void **)&d_uozp, fdm->nzpad * fdm->nxpad * sizeof(float));
	cudaMalloc((void **)&d_upzp, fdm->nzpad * fdm->nxpad * sizeof(float));	
	cudaMalloc((void **)&d_uazp, fdm->nzpad * fdm->nxpad * sizeof(float));
    cudaMalloc((void **)&d_uvzp, fdm->nzpad * fdm->nxpad * sizeof(float));

	cudaMalloc((void **)&d_umxp, fdm->nzpad * fdm->nxpad * sizeof(float));
	cudaMalloc((void **)&d_uoxp, fdm->nzpad * fdm->nxpad * sizeof(float));
	cudaMalloc((void **)&d_upxp, fdm->nzpad * fdm->nxpad * sizeof(float));
	cudaMalloc((void **)&d_uaxp, fdm->nzpad * fdm->nxpad * sizeof(float));
    cudaMalloc((void **)&d_uvxp, fdm->nzpad * fdm->nxpad * sizeof(float));
	
    cudaMalloc((void **)&d_tzzp, fdm->nzpad * fdm->nxpad * sizeof(float));
    cudaMalloc((void **)&d_tzxp, fdm->nzpad * fdm->nxpad * sizeof(float));
    cudaMalloc((void **)&d_txxp, fdm->nzpad * fdm->nxpad * sizeof(float));

	cudaMalloc((void **)&d_dzzp, fdm->nzpad * fdm->nxpad * sizeof(float));
	cudaMalloc((void **)&d_dzxp, fdm->nzpad * fdm->nxpad * sizeof(float));
	cudaMalloc((void **)&d_dxxp, fdm->nzpad * fdm->nxpad * sizeof(float));

	cudaMalloc((void **)&d_umzs, fdm->nzpad * fdm->nxpad * sizeof(float));
    cudaMalloc((void **)&d_uozs, fdm->nzpad * fdm->nxpad * sizeof(float));
    cudaMalloc((void **)&d_upzs, fdm->nzpad * fdm->nxpad * sizeof(float));
    cudaMalloc((void **)&d_uazs, fdm->nzpad * fdm->nxpad * sizeof(float));
    cudaMalloc((void **)&d_uvzs, fdm->nzpad * fdm->nxpad * sizeof(float));

    cudaMalloc((void **)&d_umxs, fdm->nzpad * fdm->nxpad * sizeof(float));
    cudaMalloc((void **)&d_uoxs, fdm->nzpad * fdm->nxpad * sizeof(float));
    cudaMalloc((void **)&d_upxs, fdm->nzpad * fdm->nxpad * sizeof(float));
    cudaMalloc((void **)&d_uaxs, fdm->nzpad * fdm->nxpad * sizeof(float));
    cudaMalloc((void **)&d_uvxs, fdm->nzpad * fdm->nxpad * sizeof(float));

    cudaMalloc((void **)&d_tzzs, fdm->nzpad * fdm->nxpad * sizeof(float));
    cudaMalloc((void **)&d_tzxs, fdm->nzpad * fdm->nxpad * sizeof(float));
    cudaMalloc((void **)&d_txxs, fdm->nzpad * fdm->nxpad * sizeof(float));

    cudaMalloc((void **)&d_dzzs, fdm->nzpad * fdm->nxpad * sizeof(float));
    cudaMalloc((void **)&d_dzxs, fdm->nzpad * fdm->nxpad * sizeof(float));
    cudaMalloc((void **)&d_dxxs, fdm->nzpad * fdm->nxpad * sizeof(float));
	
	sf_check_gpu_error("allocate grid arrays");
	
	cudaMemset(d_umzp, 0, fdm->nzpad * fdm->nxpad * sizeof(float));
	cudaMemset(d_uozp, 0, fdm->nzpad * fdm->nxpad * sizeof(float));
	cudaMemset(d_upzp, 0, fdm->nzpad * fdm->nxpad * sizeof(float));
	cudaMemset(d_uazp, 0, fdm->nzpad * fdm->nxpad * sizeof(float)); 
    cudaMemset(d_uvzp, 0, fdm->nzpad * fdm->nxpad * sizeof(float));
	
	cudaMemset(d_umxp, 0, fdm->nzpad * fdm->nxpad * sizeof(float));
	cudaMemset(d_uoxp, 0, fdm->nzpad * fdm->nxpad * sizeof(float));
	cudaMemset(d_upxp, 0, fdm->nzpad * fdm->nxpad * sizeof(float));
	cudaMemset(d_uaxp, 0, fdm->nzpad * fdm->nxpad * sizeof(float));
    cudaMemset(d_uvxp, 0, fdm->nzpad * fdm->nxpad * sizeof(float));

	cudaMemset(d_tzzp, 0, fdm->nzpad * fdm->nxpad * sizeof(float));
	cudaMemset(d_tzxp, 0, fdm->nzpad * fdm->nxpad * sizeof(float));
	cudaMemset(d_txxp, 0, fdm->nzpad * fdm->nxpad * sizeof(float));

    cudaMemset(d_dzzp, 0, fdm->nzpad * fdm->nxpad * sizeof(float));
    cudaMemset(d_dzxp, 0, fdm->nzpad * fdm->nxpad * sizeof(float));
    cudaMemset(d_dxxp, 0, fdm->nzpad * fdm->nxpad * sizeof(float));

	cudaMemset(d_umzs, 0, fdm->nzpad * fdm->nxpad * sizeof(float));
    cudaMemset(d_uozs, 0, fdm->nzpad * fdm->nxpad * sizeof(float));
    cudaMemset(d_upzs, 0, fdm->nzpad * fdm->nxpad * sizeof(float));
    cudaMemset(d_uazs, 0, fdm->nzpad * fdm->nxpad * sizeof(float));
    cudaMemset(d_uvzs, 0, fdm->nzpad * fdm->nxpad * sizeof(float));

    cudaMemset(d_umxs, 0, fdm->nzpad * fdm->nxpad * sizeof(float));
    cudaMemset(d_uoxs, 0, fdm->nzpad * fdm->nxpad * sizeof(float));
    cudaMemset(d_upxs, 0, fdm->nzpad * fdm->nxpad * sizeof(float));
    cudaMemset(d_uaxs, 0, fdm->nzpad * fdm->nxpad * sizeof(float));
    cudaMemset(d_uvxs, 0, fdm->nzpad * fdm->nxpad * sizeof(float));

    cudaMemset(d_tzzs, 0, fdm->nzpad * fdm->nxpad * sizeof(float));
    cudaMemset(d_tzxs, 0, fdm->nzpad * fdm->nxpad * sizeof(float));
    cudaMemset(d_txxs, 0, fdm->nzpad * fdm->nxpad * sizeof(float));

    cudaMemset(d_dzzs, 0, fdm->nzpad * fdm->nxpad * sizeof(float));
    cudaMemset(d_dzxs, 0, fdm->nzpad * fdm->nxpad * sizeof(float));
    cudaMemset(d_dxxs, 0, fdm->nzpad * fdm->nxpad * sizeof(float));

	sf_check_gpu_error("initialize grid arrays");

        if(opot) {
	
        cudaMalloc((void **)&d_potp, fdm->nzpad * fdm->nxpad * sizeof(float));
        cudaMalloc((void **)&d_pots, fdm->nzpad * fdm->nxpad * sizeof(float));
        cudaMemset(d_potp, 0, fdm->nzpad * fdm->nxpad * sizeof(float));
        cudaMemset(d_pots, 0, fdm->nzpad * fdm->nxpad * sizeof(float));

        }

	/*------------------------------------------------------------*/
    /* precompute 1/ro * dt^2				                      */
	/*------------------------------------------------------------*/
	dim3 dimGrid5(ceil(fdm->nxpad/16.0f),ceil(fdm->nzpad/16.0f));
	dim3 dimBlock5(16,16);
	computeRo<<<dimGrid5, dimBlock5>>>(d_ro, d_ro2, dt, fdm->nxpad, fdm->nzpad, NOP);
	sf_check_gpu_error("computeRo Kernel");
	

    /*------------------------------------------------------------*/
    /* 
     *  MAIN LOOP
     */
    /*------------------------------------------------------------*/
    if(verb) fprintf(stderr,"\n");
    for (it=nt-1; it>=0; it--) {
		if(verb) fprintf(stderr,"\b\b\b\b\b\b%d",it);

		/*------------------------------------------------------------*/
        /* from displacement to strain AND strain to stress           */
        /*      - Compute strains from displacements as in equation 1 */
        /*          - Step #1   (Steps denoted are as in Figure 2)    */
        /*      - Compute stress from strain as in equation 2         */
        /*          - Step #2                                         */
        /*------------------------------------------------------------*/
            dim3 dimGrid9(ceil(fdm->nxpad/16.0f), ceil(fdm->nzpad/16.0f));
            dim3 dimBlock9(16,16);
		dispToStrain_strainToStress<<<dimGrid9, dimBlock9>>>(d_txxp, d_tzzp, d_tzxp, d_dxxp, d_dzzp, d_dzxp, d_uoxp, d_uozp, d_c11, d_c33, d_c55, d_c13, idx, idz, fdm->nxpad, fdm->nzpad, NOP);
			dispToStrain_strainToStress<<<dimGrid9, dimBlock9>>>(d_txxs, d_tzzs, d_tzxs, d_dxxs, d_dzzs, d_dzxs, d_uoxs, d_uozs, d_c11, d_c33, d_c55, d_c13, idx, idz, fdm->nxpad, fdm->nzpad, NOP);
            sf_check_gpu_error("dispToStrainToStress Kernel");

		/*------------------------------------------------------------*/
        /* free surface boundary condition                            */
        /*      - sets the z-component of stress tensor along the     */
        /*          free surface boundary to 0                        */
        /*          - Step #3                                         */
        /*------------------------------------------------------------*/
            if(fsrf) {
                dim3 dimGrid3(ceil(fdm->nxpad/16.0f),ceil(fdm->nb/16.0f));
                dim3 dimBlock3(16,16);
                freeSurf<<<dimGrid3,dimBlock3>>>(d_tzzp, d_tzxp, fdm->nxpad, fdm->nb);
				freeSurf<<<dimGrid3,dimBlock3>>>(d_tzzs, d_tzxs, fdm->nxpad, fdm->nb);
                sf_check_gpu_error("freeSurf Kernel");
            }

		/*------------------------------------------------------------*/
        /* from stress to acceleration                                */
        /*      - Step #4                                             */
        /*------------------------------------------------------------*/
            dim3 dimGrid4(ceil((fdm->nxpad-(2*NOP))/16.0f),ceil((fdm->nzpad-(2*NOP))/16.0f));
            dim3 dimBlock4(16,16);
		stressToAcceleration<<<dimGrid4, dimBlock4>>>(d_uaxp, d_uazp, d_txxp, d_tzzp, d_tzxp, idx, idz, fdm->nxpad, fdm->nzpad);
            stressToAcceleration<<<dimGrid4, dimBlock4>>>(d_uaxs, d_uazs, d_txxs, d_tzzs, d_tzxs, idx, idz, fdm->nxpad, fdm->nzpad);
            sf_check_gpu_error("stressToAcceleration Kernel");	

		/*------------------------------------------------------------*/
        /* inject displacement data as accelaration                   */
        /*      - Step #5                                             */
        /*------------------------------------------------------------*/
            dim3 dimGrid_inject(MIN(nr,ceil(nr/1024.0f)), 1, 1);
            dim3 dimBlock_inject(MIN(nr, 1024), 1, 1);
            lint2d_inject_gpu<<<dimGrid_inject, dimBlock_inject>>>(d_dd_pz, nr, fdm->nxpad, it, d_uazp, d_Rjz, d_Rjx, d_Rw00, d_Rw01, d_Rw10, d_Rw11);
            lint2d_inject_gpu<<<dimGrid_inject, dimBlock_inject>>>(d_dd_px, nr, fdm->nxpad, it, d_uaxp, d_Rjz, d_Rjx, d_Rw00, d_Rw01, d_Rw10, d_Rw11);
            lint2d_inject_gpu<<<dimGrid_inject, dimBlock_inject>>>(d_dd_sz, nr, fdm->nxpad, it, d_uazs, d_Rjz, d_Rjx, d_Rw00, d_Rw01, d_Rw10, d_Rw11);
            lint2d_inject_gpu<<<dimGrid_inject, dimBlock_inject>>>(d_dd_sx, nr, fdm->nxpad, it, d_uaxs, d_Rjz, d_Rjx, d_Rw00, d_Rw01, d_Rw10, d_Rw11);
            sf_check_gpu_error("lint2d_inject_gpu Kernel");		

		/*------------------------------------------------------------*/
		/* step backward in time                                      */
		/*	- Compute backward time step based on acceleration     	  */
		/*		- Step #6                                     		  */
		/*------------------------------------------------------------*/
			dim3 dimGrid6(ceil(fdm->nxpad/16.0f),ceil(fdm->nzpad/12.0f));
			dim3 dimBlock6(16,12);
			stepTime<<<dimGrid6, dimBlock6>>>(d_upzp, d_uozp, d_umzp, d_uazp, d_upxp, d_uoxp, d_umxp, d_uaxp, d_ro, fdm->nxpad, fdm->nzpad);
			stepTime<<<dimGrid6, dimBlock6>>>(d_upzs, d_uozs, d_umzs, d_uazs, d_upxs, d_uoxs, d_umxs, d_uaxs, d_ro, fdm->nxpad, fdm->nzpad);
			sf_check_gpu_error("stepTime Kernel");
		
		/* circulate wavefield arrays */
		d_utzp=d_umzp; d_utxp=d_umxp;
		d_umzp=d_uozp; d_umxp=d_uoxp;
		d_uozp=d_upzp; d_uoxp=d_upxp;
		d_upzp=d_utzp; d_upxp=d_utxp;

		d_utzs=d_umzs; d_utxs=d_umxs;
        d_umzs=d_uozs; d_umxs=d_uoxs;
        d_uozs=d_upzs; d_uoxs=d_upxs;
        d_upzs=d_utzs; d_upxs=d_utxs;
	
		/*------------------------------------------------------------*/
		/* apply boundary conditions                                  */
		/*		- Step #8                                     		  */
		/*------------------------------------------------------------*/
			if(dabc) {
				
				/*---------------------------------------------------------------*/
				/* apply One-way Absorbing BC as in (Clayton and Enquist, 1977)  */
				/*---------------------------------------------------------------*/
				/* One-way Absorbing BC */
				dim3 dimGrid_TB(ceil(fdm->nxpad/192.0f), 2, 1);
				dim3 dimBlock_TB(MIN(192, fdm->nxpad), 1, 1);

				dim3 dimGrid_LR(2, ceil(fdm->nzpad/192.0f), 1);
				dim3 dimBlock_LR(1, MIN(192, fdm->nzpad), 1);

				abcone2d_apply_TB_gpu<<<dimGrid_TB, dimBlock_TB, 0, stream[0]>>>(d_uozp, d_umzp, d_bzl_s, d_bzh_s, fdm->nxpad, fdm->nzpad, fsrf);
				abcone2d_apply_LR_gpu<<<dimGrid_LR, dimBlock_LR, 0, stream[0]>>>(d_uozp, d_umzp, d_bxl_s, d_bxh_s, fdm->nxpad, fdm->nzpad);
				abcone2d_apply_TB_gpu<<<dimGrid_TB, dimBlock_TB, 0, stream[0]>>>(d_uozs, d_umzs, d_bzl_s, d_bzh_s, fdm->nxpad, fdm->nzpad, fsrf);
                abcone2d_apply_LR_gpu<<<dimGrid_LR, dimBlock_LR, 0, stream[0]>>>(d_uozs, d_umzs, d_bxl_s, d_bxh_s, fdm->nxpad, fdm->nzpad);

				abcone2d_apply_TB_gpu<<<dimGrid_TB, dimBlock_TB, 0, stream[1]>>>(d_uoxp, d_umxp, d_bzl_s, d_bzh_s, fdm->nxpad, fdm->nzpad, fsrf);
				abcone2d_apply_LR_gpu<<<dimGrid_LR, dimBlock_LR, 0, stream[1]>>>(d_uoxp, d_umxp, d_bxl_s, d_bxh_s, fdm->nxpad, fdm->nzpad);
				abcone2d_apply_TB_gpu<<<dimGrid_TB, dimBlock_TB, 0, stream[1]>>>(d_uoxs, d_umxs, d_bzl_s, d_bzh_s, fdm->nxpad, fdm->nzpad, fsrf);
                abcone2d_apply_LR_gpu<<<dimGrid_LR, dimBlock_LR, 0, stream[1]>>>(d_uoxs, d_umxs, d_bxl_s, d_bxh_s, fdm->nxpad, fdm->nzpad);
	

				/*---------------------------------------------------------------*/
				/* apply Sponge BC as in (Cerjan, et al., 1985)                  */
				/*---------------------------------------------------------------*/
				dim3 dimGrid_TB2(ceil(fdm->nxpad/256.0f), (fdm->nb * 1), 1);
				dim3 dimBlock_TB2(256,1,1);

				dim3 dimGrid_LR2(ceil(fdm->nb/256.0f), fdm->nzpad, 1);
				dim3 dimBlock_LR2(MIN(256, fdm->nb),1,1);

				sponge2d_apply_LR_gpu<<<dimGrid_LR2, dimBlock_LR2, 0, stream[2]>>>(d_upzp, d_spo, fdm->nxpad, fdm->nb, nx);
				sponge2d_apply_TB_gpu<<<dimGrid_TB2, dimBlock_TB2, 0, stream[2]>>>(d_upzp, d_spo, fdm->nxpad, fdm->nb, nz);
				sponge2d_apply_LR_gpu<<<dimGrid_LR2, dimBlock_LR2, 0, stream[2]>>>(d_upzs, d_spo, fdm->nxpad, fdm->nb, nx);
                sponge2d_apply_TB_gpu<<<dimGrid_TB2, dimBlock_TB2, 0, stream[2]>>>(d_upzs, d_spo, fdm->nxpad, fdm->nb, nz);

				sponge2d_apply_LR_gpu<<<dimGrid_LR2, dimBlock_LR2, 0, stream[3]>>>(d_upxp, d_spo, fdm->nxpad, fdm->nb, nx);
				sponge2d_apply_TB_gpu<<<dimGrid_TB2, dimBlock_TB2, 0, stream[3]>>>(d_upxp, d_spo, fdm->nxpad, fdm->nb, nz);
				sponge2d_apply_LR_gpu<<<dimGrid_LR2, dimBlock_LR2, 0, stream[3]>>>(d_upxs, d_spo, fdm->nxpad, fdm->nb, nx);
                sponge2d_apply_TB_gpu<<<dimGrid_TB2, dimBlock_TB2, 0, stream[3]>>>(d_upxs, d_spo, fdm->nxpad, fdm->nb, nz);
	
				cudaStreamSynchronize(stream[0]);
				sponge2d_apply_LR_gpu<<<dimGrid_LR2, dimBlock_LR2, 0, stream[0]>>>(d_umzp, d_spo, fdm->nxpad, fdm->nb, nx);			
				sponge2d_apply_TB_gpu<<<dimGrid_TB2, dimBlock_TB2, 0, stream[0]>>>(d_umzp, d_spo, fdm->nxpad, fdm->nb, nz);
				sponge2d_apply_LR_gpu<<<dimGrid_LR2, dimBlock_LR2, 0, stream[0]>>>(d_umzs, d_spo, fdm->nxpad, fdm->nb, nx);
                sponge2d_apply_TB_gpu<<<dimGrid_TB2, dimBlock_TB2, 0, stream[0]>>>(d_umzs, d_spo, fdm->nxpad, fdm->nb, nz);

				sponge2d_apply_LR_gpu<<<dimGrid_LR2, dimBlock_LR2, 0, stream[4]>>>(d_uozp, d_spo, fdm->nxpad, fdm->nb, nx);
				sponge2d_apply_TB_gpu<<<dimGrid_TB2, dimBlock_TB2, 0, stream[4]>>>(d_uozp, d_spo, fdm->nxpad, fdm->nb, nz);
				sponge2d_apply_LR_gpu<<<dimGrid_LR2, dimBlock_LR2, 0, stream[4]>>>(d_uozs, d_spo, fdm->nxpad, fdm->nb, nx);
                sponge2d_apply_TB_gpu<<<dimGrid_TB2, dimBlock_TB2, 0, stream[4]>>>(d_uozs, d_spo, fdm->nxpad, fdm->nb, nz);
	
				cudaStreamSynchronize(stream[1]);
				sponge2d_apply_LR_gpu<<<dimGrid_LR2, dimBlock_LR2, 0, stream[1]>>>(d_umxp, d_spo, fdm->nxpad, fdm->nb, nx);			
				sponge2d_apply_TB_gpu<<<dimGrid_TB2, dimBlock_TB2, 0, stream[1]>>>(d_umxp, d_spo, fdm->nxpad, fdm->nb, nz);
				sponge2d_apply_LR_gpu<<<dimGrid_LR2, dimBlock_LR2, 0, stream[1]>>>(d_umxs, d_spo, fdm->nxpad, fdm->nb, nx);
                sponge2d_apply_TB_gpu<<<dimGrid_TB2, dimBlock_TB2, 0, stream[1]>>>(d_umxs, d_spo, fdm->nxpad, fdm->nb, nz);
	
				sponge2d_apply_LR_gpu<<<dimGrid_LR2, dimBlock_LR2, 0, stream[5]>>>(d_uoxp, d_spo, fdm->nxpad, fdm->nb, nx);			
				sponge2d_apply_TB_gpu<<<dimGrid_TB2, dimBlock_TB2, 0, stream[5]>>>(d_uoxp, d_spo, fdm->nxpad, fdm->nb, nz);
				sponge2d_apply_LR_gpu<<<dimGrid_LR2, dimBlock_LR2, 0, stream[5]>>>(d_uoxs, d_spo, fdm->nxpad, fdm->nb, nx);
                sponge2d_apply_TB_gpu<<<dimGrid_TB2, dimBlock_TB2, 0, stream[5]>>>(d_uoxs, d_spo, fdm->nxpad, fdm->nb, nz);
		
				cudaDeviceSynchronize();

				sf_check_gpu_error("boundary condition kernels");
			}

                    if(opot) {

                /*------------------------------------------------------------*/
                /* calculate potentials                                       */
                /*      - Step #11                                            */
                /*------------------------------------------------------------*/
                        potentials_gpu<<<dimGrid9, dimBlock9>>>(d_uozp, d_uoxp, d_potp, d_pots, idx, idz, fdm->nxpad, fdm->nzpad, NOP);
                        sf_check_gpu_error("potentials_gpu Kernel");

                /*------------------------------------------------------------*/
                /* apply correlation-based imaging condition if opot=y        */
                /*      - Step #                                              */
                /*------------------------------------------------------------*/
                    if(ictype == 1) {

                        correlation_imaging_condition_gpu<<<dimGrid9, dimBlock9>>>(d_potp, d_pots, d_img, fdm->nxpad, fdm->nzpad);
                        sf_check_gpu_error("correlation_imaging_condition_gpu Kernel"); // PS crosscorrelation imaging

                    } else if(ictype == 2) {

                        correlation_imaging_condition_gpu<<<dimGrid9, dimBlock9>>>(d_potp, d_potp, d_img, fdm->nxpad, fdm->nzpad);
                        sf_check_gpu_error("correlation_imaging_condition_gpu Kernel"); // PP autocorrelation imaging

                    } else if(ictype == 3) {

                        correlation_imaging_condition_gpu<<<dimGrid9, dimBlock9>>>(d_pots, d_pots, d_img, fdm->nxpad, fdm->nzpad);
                        sf_check_gpu_error("correlation_imaging_condition_gpu Kernel"); // SS autocorrelation imaging

                    }

                /*------------------------------------------------------------*/
                /* apply extended correlation imaging condition if opot=y     */
                /*      - Step #                                              */
                /*------------------------------------------------------------*/
                    if(eictype == 1) {
                        dim3 dimGrid7(ceil((2*nhx+1)/16.0f),ceil((2*nhz+1)/16.0f));
                        dim3 dimBlock7(16,16);
                        extended_correlation_imaging_condition_gpu<<<dimGrid7, dimBlock7>>>(d_potp, d_pots, d_eimg, nhx, nhz, cipx, cipz, fdm->nxpad, fdm->nb);
                        sf_check_gpu_error("extended_correlation_imaging_condition_gpu Kernel"); // PS conventional extended imaging

                    }

                /*------------------------------------------------------------*/
                /* cut wavefield and save                                     */
                /*              - Step #                                      */
                /*------------------------------------------------------------*/
                    if(snap && it%jsnap==0) {
                                cudaMemcpy(h_uox, d_pots, fdm->nzpad * fdm->nxpad * sizeof(float), cudaMemcpyDeviceToHost);
                                cudaMemcpy(h_uoz, d_potp, fdm->nzpad * fdm->nxpad * sizeof(float), cudaMemcpyDeviceToHost);

                                for (int x = 0; x < fdm->nxpad; x++){
                                        for (int z = 0; z < fdm->nzpad; z++){
                                                uox[x][z] = h_uox[z * fdm->nxpad + x];
                                                uoz[x][z] = h_uoz[z * fdm->nxpad + x];
                                        }
                                }

                                cut2d(uoz,uc,fdm,acz,acx);
                                sf_floatwrite(uc[0],sf_n(acz)*sf_n(acx),Fwfl);

                                cut2d(uox,uc,fdm,acz,acx);
                                sf_floatwrite(uc[0],sf_n(acz)*sf_n(acx),Fwfl);
                    }

                    } else if(!opot) {

                /*------------------------------------------------------------*/
                /* apply correlation-based imaging condition if opot=n        */
                /*      - Step #                                              */
                /*------------------------------------------------------------*/
                    if(ictype == 1) {

                        correlation_imaging_condition_gpu<<<dimGrid9, dimBlock9>>>(d_uozp, d_uoxs, d_img, fdm->nxpad, fdm->nzpad);
                        sf_check_gpu_error("correlation_imaging_condition_gpu Kernel"); // PS crosscorrelation imaging

                    } else if(ictype == 2) {

                        correlation_imaging_condition_gpu<<<dimGrid9, dimBlock9>>>(d_uozp, d_uozp, d_img, fdm->nxpad, fdm->nzpad);
                        sf_check_gpu_error("correlation_imaging_condition_gpu Kernel"); // PP autocorrelation imaging

                    } else if(ictype == 3) {

                        correlation_imaging_condition_gpu<<<dimGrid9, dimBlock9>>>(d_uoxs, d_uoxs, d_img, fdm->nxpad, fdm->nzpad);
                        sf_check_gpu_error("correlation_imaging_condition_gpu Kernel"); // SS autocorrelation imaging

                    } else if(ictype == 4) {

						/*------------------------------------------------------------*/
						/* Energy-norm imaging condition    						  */
						/*------------------------------------------------------------*/
						first_order_time_derivative_gpu<<<dimGrid9, dimBlock9>>>(d_uozp, d_uoxp, d_upzp, d_upxp, d_uvzp, d_uvxp, fdm->nxpad, fdm->nzpad, dt);
						first_order_time_derivative_gpu<<<dimGrid9, dimBlock9>>>(d_uozs, d_uoxs, d_upzs, d_upxs, d_uvzs, d_uvxs, fdm->nxpad, fdm->nzpad, dt);
						sf_check_gpu_error("first_order_time_derivative_gpu Kernel");
						if(it*dt < tmax && it*dt >= tmin) {
							idxt2 = it - tmin/dt;
							energy_imaging_condition_ps_gpu<<<dimGrid9, dimBlock9>>>(d_uvzp, d_uvxp, d_uvzs, d_uvxs, d_txxp, d_tzzp, d_tzxp, d_txxs, d_tzzs, d_tzxs, d_dxxp, d_dzzp, d_dzxp, d_dxxs, d_dzzs, d_dzxs, d_ro2, d_img, fdm->nxpad, fdm->nzpad, crealf(tap_h[idxt2]));
							sf_check_gpu_error("energy_imaging_condition_ps_gpu Kernel"); // PS energy-norm imaging
						}

                    } else if(ictype == 5) {

						/*------------------------------------------------------------*/
						/*  Kinetic energy IC 			  						      */
						/*------------------------------------------------------------*/
						first_order_time_derivative_gpu<<<dimGrid9, dimBlock9>>>(d_uozp, d_uoxp, d_upzp, d_upxp, d_uvzp, d_uvxp, fdm->nxpad, fdm->nzpad, dt);
						first_order_time_derivative_gpu<<<dimGrid9, dimBlock9>>>(d_uozs, d_uoxs, d_upzs, d_upxs, d_uvzs, d_uvxs, fdm->nxpad, fdm->nzpad, dt);
						sf_check_gpu_error("first_order_time_derivative_gpu Kernel");
						if(it*dt < tmax && it*dt >= tmin) {
							idxt2 = it - tmin/dt;
							kinetic_imaging_condition_gpu<<<dimGrid9, dimBlock9>>>(d_uvzp, d_uvxp, d_uvzs, d_uvxs, d_txxp, d_tzzp, d_tzxp, d_txxs, d_tzzs, d_tzxs, d_dxxp, d_dzzp, d_dzxp, d_dxxs, d_dzzs, d_dzxs, d_ro2, d_img, fdm->nxpad, fdm->nzpad, crealf(tap_h[idxt2]));
							sf_check_gpu_error("kinetic_imaging_condition_gpu Kernel"); // PS energy-norm imaging
						}
						
                    } else if(ictype == 6) {

						/*------------------------------------------------------------*/
						/*  Potential energy IC 		  						      */
						/*------------------------------------------------------------*/
						first_order_time_derivative_gpu<<<dimGrid9, dimBlock9>>>(d_uozp, d_uoxp, d_upzp, d_upxp, d_uvzp, d_uvxp, fdm->nxpad, fdm->nzpad, dt);
						first_order_time_derivative_gpu<<<dimGrid9, dimBlock9>>>(d_uozs, d_uoxs, d_upzs, d_upxs, d_uvzs, d_uvxs, fdm->nxpad, fdm->nzpad, dt);
						sf_check_gpu_error("first_order_time_derivative_gpu Kernel");
						if(it*dt < tmax && it*dt >= tmin) {
							idxt2 = it - tmin/dt;
							potential_imaging_condition_gpu<<<dimGrid9, dimBlock9>>>(d_uvzp, d_uvxp, d_uvzs, d_uvxs, d_txxp, d_tzzp, d_tzxp, d_txxs, d_tzzs, d_tzxs, d_dxxp, d_dzzp, d_dzxp, d_dxxs, d_dzzs, d_dzxs, d_ro2, d_img, fdm->nxpad, fdm->nzpad, crealf(tap_h[idxt2]));
							sf_check_gpu_error("potential_imaging_condition_ps_gpu Kernel"); // PS energy-norm imaging
						}
						
    				} else if(ictype == 7) {

						/*------------------------------------------------------------*/
						/*  Energy Norm (Hamiltonian)   						      */
						/*------------------------------------------------------------*/
						first_order_time_derivative_gpu<<<dimGrid9, dimBlock9>>>(d_uozp, d_uoxp, d_upzp, d_upxp, d_uvzp, d_uvxp, fdm->nxpad, fdm->nzpad, dt);
						first_order_time_derivative_gpu<<<dimGrid9, dimBlock9>>>(d_uozs, d_uoxs, d_upzs, d_upxs, d_uvzs, d_uvxs, fdm->nxpad, fdm->nzpad, dt);
						sf_check_gpu_error("first_order_time_derivative_gpu Kernel");
						if(it*dt < tmax && it*dt >= tmin) {
							idxt2 = it - tmin/dt;
							hamiltonian_imaging_condition_gpu<<<dimGrid9, dimBlock9>>>(d_uvzp, d_uvxp, d_uvzs, d_uvxs, d_txxp, d_tzzp, d_tzxp, d_txxs, d_tzzs, d_tzxs, d_dxxp, d_dzzp, d_dzxp, d_dxxs, d_dzzs, d_dzxs, d_ro2, d_img, fdm->nxpad, fdm->nzpad, crealf(tap_h[idxt2]));
							sf_check_gpu_error("Hamiltonian_imaging_condition_ps_gpu Kernel"); // PS energy-norm imaging
						}

    				} else if(ictype == 8) {

						/*------------------------------------------------------------*/
						/*  Energy Ratio (Laplacian/Hamiltonian)   				      */
						/*------------------------------------------------------------*/
						first_order_time_derivative_gpu<<<dimGrid9, dimBlock9>>>(d_uozp, d_uoxp, d_upzp, d_upxp, d_uvzp, d_uvxp, fdm->nxpad, fdm->nzpad, dt);
						first_order_time_derivative_gpu<<<dimGrid9, dimBlock9>>>(d_uozs, d_uoxs, d_upzs, d_upxs, d_uvzs, d_uvxs, fdm->nxpad, fdm->nzpad, dt);
						sf_check_gpu_error("first_order_time_derivative_gpu Kernel");
						if(it*dt < tmax && it*dt >= tmin) {
							idxt2 = it - tmin/dt;
							energy_ratio_imaging_condition_gpu<<<dimGrid9, dimBlock9>>>(d_uvzp, d_uvxp, d_uvzs, d_uvxs, d_txxp, d_tzzp, d_tzxp, d_txxs, d_tzzs, d_tzxs, d_dxxp, d_dzzp, d_dzxp, d_dxxs, d_dzzs, d_dzxs, d_ro2, d_img, fdm->nxpad, fdm->nzpad, crealf(tap_h[idxt2]));
							sf_check_gpu_error("energy_ratio_imaging_condition_gpu Kernel"); // PS energy-norm imaging
						}
    				}


                   if(eictype == 2) {
                /*------------------------------------------------------------*/
                /* apply first-order time derivative and extended energy      */
                /* imaging condition if opot=n                                */
                /*      - Step #                                              */
                /*------------------------------------------------------------*/
                        //first_order_time_derivative_gpu<<<dimGrid9, dimBlock9>>>(d_uoz, d_uox, d_upz, d_upx, d_uvz, d_uvx, fdm->nxpad, fdm->nzpad, dt);
                        //sf_check_gpu_error("first_order_time_derivative_gpu Kernel");
                        //dim3 dimGrid7(ceil((2*nhx+1)/16.0f),ceil((2*nhz+1)/16.0f));
                        //dim3 dimBlock7(16,16);
						dim3 dimGrid7(ceil((2*nhx+1)/25.0f),(2*nhz+1));
                        dim3 dimBlock7(25,1);
                        extended_energy_imaging_condition_ps3_gpu<<<dimGrid7, dimBlock7>>>(d_uvzp, d_uvxp, d_uvzs, d_uvxs, d_dxxp, d_dzzp, d_dzxp, d_dxxs, d_dzzs, d_dzxs, d_c11, d_c33, d_c55, d_c13, d_ro2, d_eimg, nhx, nhz, cipx, cipz, fdm->nxpad, fdm->nzpad, fdm->nb);
						//extended_energy_imaging_condition_ps2_gpu<<<dimGrid7, dimBlock7>>>(d_uvzp, d_uvxp, d_uvzs, d_uvxs, d_dxxp, d_dzzp, d_dzxp, d_dxxs, d_dzzs, d_dzxs, d_c11, d_c33, d_c55, d_c13, d_ro2, d_eimg, nhx, nhz, cipx, cipz, ncip, fdm->nxpad, fdm->nzpad, fdm->nb);
                        sf_check_gpu_error("extended_energy_imaging_condition_ps2_gpu Kernel"); // extended energy-norm imaging

                    }

                /*------------------------------------------------------------*/
                /* cut wavefield and save                                     */
                /*              - Step #                                      */
                /*------------------------------------------------------------*/
		    if(snap && it%jsnap==0) {
				cudaMemcpy(h_uox, d_uozp, fdm->nzpad * fdm->nxpad * sizeof(float), cudaMemcpyDeviceToHost); // d_uvxp -> for horizontal P-wave particle velocity output
				cudaMemcpy(h_uoz, d_uozs, fdm->nzpad * fdm->nxpad * sizeof(float), cudaMemcpyDeviceToHost); // d_uvzp -> for vertical   P-wave particle velocity output

				for (int x = 0; x < fdm->nxpad; x++){
					for (int z = 0; z < fdm->nzpad; z++){
						uox[x][z] = h_uox[z * fdm->nxpad + x];
						uoz[x][z] = h_uoz[z * fdm->nxpad + x];
					}
				}

				cut2d(uoz,uc,fdm,acz,acx);
				sf_floatwrite(uc[0],sf_n(acz)*sf_n(acx),Fwfl);

				cut2d(uox,uc,fdm,acz,acx);
				sf_floatwrite(uc[0],sf_n(acz)*sf_n(acx),Fwfl);
		    }

            }

    }   
    
    /* . . extract conventional image from device to host . . */
    cudaMemcpy(h_img, d_img, fdm->nzpad * fdm->nxpad * sizeof(float), cudaMemcpyDeviceToHost);
    /* . . put into conventional image volume . . */
    for (int x = 0; x < fdm->nxpad; x++){
             for (int z = 0; z < fdm->nzpad; z++){
                      uoz[x][z] = h_img[z * fdm->nxpad + x];
             }
    }
    /* . . write conventional image into file . . */
    cut2d(uoz,uc,fdm,aiz,aix);
    sf_floatwrite(uc[0],sf_n(aiz)*sf_n(aix),Fimg);

    /* . . extract extended image from device to host . . */
    cudaMemcpy(h_eimg, d_eimg, sf_n(alz) * sf_n(alx) * sf_n(ahz) * sf_n(ahx) * sizeof(float), cudaMemcpyDeviceToHost);
    /* . . put into extended image volume . . */
	for (int x = 0; x < 2*nhx+1; x++){
		for (int z = 0; z < 2*nhz+1; z++){
			for (int lx = 0; lx < nlx; lx++){
				for (int lz = 0; lz < nlz; lz++){
				uez[x][z][lx][lz] = h_eimg[lz * sf_n(alx) * sf_n(ahz) * sf_n(ahx) + lx * sf_n(ahz) * sf_n(ahx) + z * sf_n(ahx) + x];
				}
			}
		}
	}
    /* . . write extended image into file . . */
    //sf_floatwrite(uez[0][0],sf_n(ahc)*sf_n(ahz)*sf_n(ahx),Feimg);
	sf_floatwrite(uez[0][0][0],sf_n(alz)*sf_n(alx)*sf_n(ahz)*sf_n(ahx),Feimg);

    if(verb) fprintf(stderr,"\n");
    
    /*------------------------------------------------------------*/
    /* deallocate host arrays */

    free(rr);
	
	if (snap){
		free(*uoz); free(uoz);
	    free(*uox); free(uox);
	    free(h_uoz);  
		free(h_uox);
		free(*uc);  free(uc);    
	}
	
	free(h_c11); free(h_c33); free(h_c55); free(h_c13); free(h_ro);
	free(h_bell);

	/*------------------------------------------------------------*/
    /* deallocate GPU arrays */

	cudaFree(d_ro);
	cudaFree(d_c11); 	cudaFree(d_c33); 	cudaFree(d_c55); 	cudaFree(d_c13);
	cudaFree(d_umzp); 	cudaFree(d_uozp); 	cudaFree(d_upzp); 	cudaFree(d_uazp); 	cudaFree(d_utzp);	cudaFree(d_uvzp);
	cudaFree(d_umxp); 	cudaFree(d_uoxp); 	cudaFree(d_upxp); 	cudaFree(d_uaxp); 	cudaFree(d_utxp);	cudaFree(d_uvxp);
	cudaFree(d_tzzp); 	cudaFree(d_tzxp); 	cudaFree(d_txxp);
	cudaFree(d_umzs);    cudaFree(d_uozs);    cudaFree(d_upzs);    cudaFree(d_uazs);    cudaFree(d_utzs);	cudaFree(d_uvzs);
    cudaFree(d_umxs);    cudaFree(d_uoxs);    cudaFree(d_upxs);    cudaFree(d_uaxs);    cudaFree(d_utxs);	cudaFree(d_uvxs);
    cudaFree(d_tzzs);    cudaFree(d_tzxs);    cudaFree(d_txxs);
	cudaFree(d_Rjz);	cudaFree(d_Rjx);
	cudaFree(d_bell);

	if (dabc){
		cudaFree(d_bzl_s); 	cudaFree(d_bzh_s);
		cudaFree(d_bxl_s); 	cudaFree(d_bxh_s);
		cudaFree(d_spo);
	}

    sf_close();
	exit(0);
}

