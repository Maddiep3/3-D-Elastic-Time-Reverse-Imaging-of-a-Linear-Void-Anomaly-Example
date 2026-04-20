/* generates an Ormsby filter */
#include<rsf.h>
#include <stdio.h>

float sincf(float x); /* calculates sin(x)/x */

/* find the maximum value of a vector */
float maxval(float *vec, int size);

int main(int argc, char* argv[]){

int nt, i, kt;
float pif1, pif1t, ormf1;
float pif2, pif2t, ormf2;
float pif3, pif3t, ormf3;
float pif4, pif4t, ormf4;
float f1,f2,f3,f4;
float dt, t;
float *ormsby;
const float PI_F = 3.141592653589793;

sf_file Fw=NULL; /* ormsby wavelet */
sf_axis at;

/* init RSF */
sf_init(argc,argv);

/* output file */
Fw = sf_output("out");

/* get parameters */
if(! sf_getint("kt",&kt)) kt=1;
if(! sf_getint("nt",&nt)) nt=1001;
if(! sf_getfloat("dt",&dt)) dt=.001;
if(! sf_getfloat("f1",&f1)) f1=5;
if(! sf_getfloat("f2",&f2)) f2=10;
if(! sf_getfloat("f3",&f3)) f3=40;
if(! sf_getfloat("f4",&f4)) f4=45;

/* output axis */
at=sf_maxa(nt,-(kt-1)*dt,dt);
sf_oaxa(Fw,at,1);

/* allocate the filter */
ormsby = sf_floatalloc(nt);

for (i = 0; i < nt; i++){
    t = (-(((float)kt-1.0)*dt)) + (((float)i-1)*dt);

    pif1 = PI_F * f1;
    pif2 = PI_F * f2;
    pif3 = PI_F * f3;
    pif4 = PI_F * f4;

    pif1t = pif1 * t;
    pif2t = pif2 * t;
    pif3t = pif3 * t;
    pif4t = pif4 * t;

    ormf1 = ((pif1*pif1) / (pif2 - pif1)) * (sincf(pif1t)*sincf(pif1t));
    ormf2 = ((pif2*pif2) / (pif2 - pif1)) * (sincf(pif2t)*sincf(pif2t));
    ormf3 = ((pif3*pif3) / (pif4 - pif3)) * (sincf(pif3t)*sincf(pif3t));
    ormf4 = ((pif4*pif4) / (pif4 - pif3)) * (sincf(pif4t)*sincf(pif4t));

    ormsby[i] = (ormf4 - ormf3) - (ormf2 - ormf1);
}

/* normalize the wavelet */
float maxOrm = maxval(ormsby, nt);
sf_warning("%f",maxOrm);
for (i = 0; i < nt; i++){
    ormsby[i] = ormsby[i] / maxOrm;
}

/* write to file */
sf_floatwrite(ormsby,nt,Fw);

/* free */
free(ormsby);

}

/* sin(x) / x*/
float sincf(float x){
    float eps = 2.2204e-16;
    float sinc;

    if (fabsf(x) <= eps){
        sinc = 1.0;
    }else {
        sinc = sinf(x) / x;
    }
    return sinc;
}

/* max val of a vector */
float maxval(float *vec, int size){
    int i;
    float max;
    
    max = vec[0];
    for (i=1; i < size; i++){
       if (vec[i] > max){
            max = vec[i];
       }          
    }
    return max;
}
