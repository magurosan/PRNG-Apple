/*
   AdvSIMD optimized MT19937-64 for Apple Sillicon(A13/M1 or greater).
   Copyright (C) 2020 Masaki Ota. All rights reserved.  

   USAGE:
    % clang -O3 mt19937-64-neon.c -o neon_test 
    % ./neon_test bench      
    100000000 random numbers generation time                                                                                 
    Original:       178.26 ms
    NEON:           145.24 ms
    NEON x2:         95.45 ms
    NEON ARRAY:      84.76 ms

    % clang -O3 -march=armv8.4-a+simd+sha3 -DHAVE_SHA3 mt19937-64-neon.c -o m1test 
    % ./m1test bench 
    100000000 random numbers generation time                                               
    Original:            173.37 ms
    NEON+SHA3:           111.67 ms
    NEON+SHA3 x2:         86.29 ms
    NEON+SHA3 ARRAY:      69.13 ms
*/
/* 
   A C-program for MT19937-64 (2014/2/23 version).
   Coded by Takuji Nishimura and Makoto Matsumoto.

   This is a 64-bit version of Mersenne Twister pseudorandom number
   generator.

   Before using, initialize the state by using init_genrand64(seed)  
   or init_by_array64(init_key, key_length).

   Copyright (C) 2004, 2014, Makoto Matsumoto and Takuji Nishimura,
   All rights reserved.                          

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.

     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.

     3. The names of its contributors may not be used to endorse or promote 
        products derived from this software without specific prior written 
        permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

   References:
   T. Nishimura, ``Tables of 64-bit Mersenne Twisters''
     ACM Transactions on Modeling and 
     Computer Simulation 10. (2000) 348--357.
   M. Matsumoto and T. Nishimura,
     ``Mersenne Twister: a 623-dimensionally equidistributed
       uniform pseudorandom number generator''
     ACM Transactions on Modeling and 
     Computer Simulation 8. (Jan. 1998) 3--30.

   Any feedback is very welcome.
   http://www.math.hiroshima-u.ac.jp/~m-mat/MT/emt.html
   email: m-mat @ math.sci.hiroshima-u.ac.jp (remove spaces)
*/
#include <stdint.h>
#include <arm_neon.h>

#define NN 312
#define MM 156
#define MATRIX_A UINT64_C(0xB5026F5AA96619E9)
#define UM UINT64_C(0xFFFFFFFF80000000) /* Most significant 33 bits */
#define LM UINT64_C(0x7FFFFFFF) /* Least significant 31 bits */


#if defined(__ARM_FEATURE_SHA3)
#define HAVE_SHA3
#define EOR3(a,b,c) veor3q_u64(a,b,c)
#define BCAX(a,b,c) vbcaxq_u64(a,b,c)

#elif defined(__apple_build_version__) && defined(HAVE_SHA3)
/*
 * SHA3 missing intrinsics
 */
uint64x2_t veor3q_u64_wrapper(uint64x2_t a, uint64x2_t b, uint64x2_t c)
{
    uint64x2_t r;
    __asm__ ("eor3.16b %0,%1,%2,%3" :"=w"(r) :"w"(a), "w"(b), "w"(c));
    return r;
}
uint64x2_t vbcaxq_u64_wrapper(uint64x2_t a, uint64x2_t b, uint64x2_t c)
{
    uint64x2_t r;
    __asm__ ("bcax.16b %0,%1,%2,%3" :"=w"(r) :"w"(a), "w"(b), "w"(c));
    return r;
}
#define EOR3(a,b,c) veor3q_u64_wrapper(a,b,c)
#define BCAX(a,b,c) vbcaxq_u64_wrapper(a,b,c)

#else // NO SHA3
/* emulate SHA3 */
#define EOR3(a,b,c) veorq_u64(a, veorq_u64(b,c)) 
#define BCAX(a,b,c) veorq_u64(a, vbicq_u64(b,c))
#endif

#if defined(__apple_build_version__) 
#define INSERT_IF_TRUE(a,b,mask)  __asm__ ("bit.16b %1,%2,%3" :"=w"(a) : "w"(a), "w"(b), "w"(mask))
#else 
#define INSERT_IF_TRUE(a,b,mask)  a = vbslq_u64(mask, b, a)
#endif

/* The array for the state vector */
static uint64x2_t mt_v[NN/2]; 
static uint64_t* const mt = (uint64_t* const)mt_v; 

/* mti==NN+1 means mt[NN] is not initialized */
static int mti=NN+1;  

/* initializes mt[NN] with a seed */
void init_genrand64(uint64_t seed)
{
    mt[0] = seed;
    for (mti=1; mti<NN; mti++) 
        mt[mti] =  (UINT64_C(6364136223846793005) * (mt[mti-1] ^ (mt[mti-1] >> 62)) + mti);
}

/* initialize by an array with array-length */
/* init_key is the array for initializing keys */
/* key_length is its length */
void init_by_array64(uint64_t init_key[], uint64_t key_length)
{
    unsigned int i, j;
    uint64_t k;
    init_genrand64(UINT64_C(19650218));
    i=1; j=0;
    k = (NN>key_length ? NN : key_length);
    for (; k; k--) {
        mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 62)) * UINT64_C(3935559000370003845)))
          + init_key[j] + j; /* non linear */
        i++; j++;
        if (i>=NN) { mt[0] = mt[NN-1]; i=1; }
        if (j>=key_length) j=0;
    }
    for (k=NN-1; k; k--) {
        mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 62)) * UINT64_C(2862933555777941757)))
          - i; /* non linear */
        i++;
        if (i>=NN) { mt[0] = mt[NN-1]; i=1; }
    }

    mt[0] = UINT64_C(1) << 63; /* MSB is 1; assuring non-zero initial array */ 
}


/* generates a random number on [0, 2^64-1]-interval */
static __inline uint64_t genrand64_int64_original(void)
{
    static uint64_t mag01[2]={0ULL, MATRIX_A};
    int i;
    unsigned long long x;

    if (mti >= NN) { /* generate NN words at one time */

        /* if init_genrand64() has not been called, */
        /* a default initial seed is used     */
        if (mti == NN+1) 
            init_genrand64(5489ULL); 

        for (i=0;i<NN-MM;i++) {
            x = (mt[i]&UM)|(mt[i+1]&LM);
            mt[i] = mt[i+MM] ^ (x>>1) ^ mag01[(int)(x&1ULL)];
        }
        for (;i<NN-1;i++) {
            x = (mt[i]&UM)|(mt[i+1]&LM);
            mt[i] = mt[i+(MM-NN)] ^ (x>>1) ^ mag01[(int)(x&1ULL)];
        }
        x = (mt[NN-1]&UM)|(mt[0]&LM);
        mt[NN-1] = mt[MM-1] ^ (x>>1) ^ mag01[(int)(x&1ULL)];

        mti = 0;
    }
  
    x = mt[mti++];

    x ^= (x >> 29) & 0x5555555555555555ULL;
    x ^= (x << 17) & 0x71D67FFFEDA60000ULL;
    x ^= (x << 37) & 0xFFF7EEE000000000ULL;
    x ^= (x >> 43);

    return x;
}


static __inline void update_neon_sha3() {
    int i;
    uint64x2_t y, z, next;
    static uint64x2_t mtx = { MATRIX_A, MATRIX_A };
    static uint64x2_t mask = { 0x7FFFFFFF, 0x7FFFFFFF };

#define MAG(x) vbicq_u64(mtx,vsubq_u64(vandq_u64(x,vdupq_n_u64(1)),vdupq_n_u64(1)))
#define ALIGN(x,y,N) vreinterpretq_u64_u8(vextq_u8(vreinterpretq_u8_u64(x),vreinterpretq_u8_u64(y),N))

    y = vld1q_u64(&mt[0]);
    for (i=0; i<NN-MM; i+=2) {
        next = vld1q_u64(&mt[i+2]);
        z = ALIGN(y, next, 8);
        
        INSERT_IF_TRUE(y, z, mask);
        
        z = EOR3(vld1q_u64(&mt[i+MM]), vshrq_n_u64(y, 1), MAG(z));
        vst1q_u64(&mt[i], z);
        y = next;
    }
    for (;i<NN-2;i+=2) {
        next = vld1q_u64(&mt[i+2]);
        z = ALIGN(y, next, 8);
        
        INSERT_IF_TRUE(y, z, mask); 
        
        z = EOR3(vld1q_u64(&mt[i+MM-NN]), vshrq_n_u64(y, 1), MAG(z));
        vst1q_u64(&mt[i], z);
        y = next;
    }
    next = vld1q_u64(&mt[0]);
    z = ALIGN(y, next, 8);
    
    INSERT_IF_TRUE(y, z, mask); 
    
    z = EOR3(vld1q_u64(&mt[MM-2]), vshrq_n_u64(y, 1),  MAG(z));
    vst1q_u64(&mt[i], z);

#undef ALIGN
#undef MAG
}


/* generates a random number on [0, 2^64-1]-interval */
__inline uint64_t genrand64_int64(void)
{
    if (mti >= NN) {
        /* if init_genrand64() has not been called, */
        /* a default initial seed is used     */
        if (mti == NN+1) init_genrand64(5489ULL); 

        update_neon_sha3(); /* generate NN words at one time */
        mti = 0;
    }

#if defined(HAVE_SHA3)
    uint64x2_t wx = vcombine_u64(vld1_u64(&mt[mti]), vdup_n_u64(0));
    mti++;

    wx = BCAX(wx, vshrq_n_u64(wx, 29), vdupq_n_u64(~0x5555555555555555));
    wx = BCAX(wx, vshlq_n_u64(wx, 17), vdupq_n_u64(~0x71D67FFFEDA60000));
    wx = BCAX(wx, vshlq_n_u64(wx, 37), vdupq_n_u64(~0xFFF7EEE000000000));
    uint64_t x = vgetq_lane_u64(wx, 0);
    return x ^ (x >> 43);
#else 
    uint64_t x = mt[mti++];

    x ^= (x >> 29) & 0x5555555555555555ULL;
    x ^= (x << 17) & 0x71D67FFFEDA60000ULL;
    x ^= (x << 37) & 0xFFF7EEE000000000ULL;
    return x ^ (x >> 43);
#endif
}

/* generates a random number on [0, 2^63-1]-interval */
__inline int64_t genrand64_int63(void)
{
    return (int64_t)(genrand64_int64() >> 1);
}

/* generates a random number on [0,1]-real-interval */
__inline double genrand64_real1(void)
{
    return (genrand64_int64() >> 11) * (1.0/9007199254740991.0);
}

/* generates a random number on [0,1)-real-interval */
__inline double genrand64_real2(void)
{
    return (genrand64_int64() >> 11) * (1.0/9007199254740992.0);
}

/* generates a random number on (0,1)-real-interval */
__inline double genrand64_real3(void)
{
    return ((genrand64_int64() >> 12) + 0.5) * (1.0/4503599627370496.0);
}

static __inline uint64x2_t tempering_x2(uint64x2_t wx)
{
    wx = BCAX(wx, vshrq_n_u64(wx, 29), vdupq_n_u64(~0x5555555555555555));
    wx = BCAX(wx, vshlq_n_u64(wx, 17), vdupq_n_u64(~0x71D67FFFEDA60000));
    wx = BCAX(wx, vshlq_n_u64(wx, 37), vdupq_n_u64(~0xFFF7EEE000000000));
    wx = veorq_u64(wx, vshrq_n_u64(wx, 43));
    return wx;
}


/* generates two random numbers on [0, 2^64-1]-interval */
__inline uint64x2_t genrand64_int64_x2(void)
{
    uint64x2_t wx;
    if (mti & 1) {
        if (mti >= NN+1) {
            init_genrand64(5489ULL); 
        } else {
            mti +=1;
        }
    }    
    if (mti >= NN) {
        update_neon_sha3();
        mti = 0;
    }
    wx = vld1q_u64(&mt[mti]); 
    mti += 2;

    return tempering_x2(wx);
}

/* 
 * (experimental)
 *  generates bulked random numbers on [0, 2^64-1]-interval 
 */
void genrand64_int64_array(uint64_t *array, int length)
{
    uint64x2_t* array_v = (uint64x2_t*)array;

    while (length >= NN) {
        update_neon_sha3();
        for (int i = 0; i < NN/2; i++) array_v[i] = tempering_x2(mt_v[i]); 
        array_v += NN/2;
        length -= NN;
    }
    update_neon_sha3();
    for (int i = 0; i < length/2; i++) array_v[i] = tempering_x2(mt_v[i]); 
    mti = length;
}

/* generates two random numbers on [0, 2^63-1]-interval */
__inline uint64x2_t genrand64_int63_x2(void)
{
    return vshrq_n_u64(genrand64_int64_x2(), 1);
}

/* generates two random number on [0,1]-real-interval */
__inline float64x2_t genrand64_real1_x2(void)
{
    uint64x2_t u = genrand64_int64_x2();
    u = vshrq_n_u64(u, 11);
    float64x2_t v = vcvtq_u64_f64(u);
    return vmulq_f64(v, vdupq_n_f64(1.0/9007199254740991.0));
}

/* generates two random numbers on [0,1]-real-interval */
__inline float64x2_t genrand64_real2_x2(void)
{
    uint64x2_t u = genrand64_int64_x2();
    u = vshrq_n_u64(u, 11);
    float64x2_t v = vcvtq_u64_f64(u);
    return vmulq_f64(v, vdupq_n_f64(1.0/9007199254740992.0));
}

/* generates two random numbers on (0,1)-real-interval */
__inline float64x2_t genrand64_real3_x2(void)
{
    uint64x2_t u = genrand64_int64_x2();
    u = vshrq_n_u64(u, 12);
    float64x2_t v = vcvtq_u64_f64(u);
    return vfmaq_f64(vdupq_n_f64(1.0/9007199254740992.0), 
        v, vdupq_n_f64(1.0/4503599627370496.0));
}

/*----------------------------------------------------------------*/
/* Test Program                                                   */
/*----------------------------------------------------------------*/
#include <stdio.h>
#include <string.h>
#include <time.h>

#define COUNT  100000000

double bench_original()
{
    uint64_t best = -1;
    for (int t = 0; t < 10; t++) {
        clock_t begin = clock();
        uint64_t result = 0;
        volatile uint64_t result2 = 0;
        /* here, do your time-consuming job */
        for (int i = 0; i < COUNT; i++) {
            result += genrand64_int64_original();
        }
        result2 = result;
        clock_t end = clock();
        uint64_t time = end - begin;
        if (best > time) best = time;
    }
    return (double) best / CLOCKS_PER_SEC;
}

double bench1()
{
    uint64_t best = -1;
    for (int t = 0; t < 10; t++) {
        clock_t begin = clock();
        uint64_t result = 0;
        volatile uint64_t result2 = 0;
        /* here, do your time-consuming job */
        for (int i = 0; i < COUNT; i++) {
            result += genrand64_int64();
        }
        result2 = result;
        clock_t end = clock();
        uint64_t time = end - begin;
        if (best > time) best = time;
    }
    return (double) best / CLOCKS_PER_SEC;
}

double bench2()
{
    uint64_t best = -1;
    for (int t = 0; t < 10; t++) {
        clock_t begin = clock();
        uint64_t result = 0;
        volatile uint64_t result2 = 0;
        /* here, do your time-consuming job */
        for (int i = 0; i < COUNT/2; i++) {
            uint64x2_t res = genrand64_int64_x2();
            result += res[0] + res[1];
        }
        result2 = result;
        clock_t end = clock();
        uint64_t time = end - begin;
        if (best > time) best = time;
    }
    return (double) best / CLOCKS_PER_SEC;
}

#define ARRAY_SIZE 100000
static uint64_t array[ARRAY_SIZE];
double bench_array()
{
    uint64_t best = -1;
    for (int t = 0; t < 10; t++) {
        clock_t begin = clock();

        uint64_t result = 0;
        volatile uint64_t result2 = 0;

        /* here, do your time-consuming job */        
        for (int i = 0; i < COUNT/ARRAY_SIZE; i++) {
            genrand64_int64_array(array, ARRAY_SIZE);
            result ^= array[0];
        }
        result2 = result;

        clock_t end = clock();
        uint64_t time = end - begin;
        if (best > time) best = time;
    }
    return (double) best / CLOCKS_PER_SEC;
}

int main(int argc, char* argv[])
{
    if (argc >= 2 && strcmp(argv[1], "bench") == 0){
        printf("%d random numbers generation time\n", COUNT);
#if defined(HAVE_SHA3)
        printf("Original:        %10.2f ms\n", 1000*bench_original());
        printf("NEON+SHA3:       %10.2f ms\n", 1000*bench1());
        printf("NEON+SHA3 x2:    %10.2f ms\n", 1000*bench2());
        printf("NEON+SHA3 ARRAY: %10.2f ms\n", 1000*bench_array());
#else
        printf("Original:   %10.2f ms\n", 1000*bench_original());
        printf("NEON:       %10.2f ms\n", 1000*bench1());
        printf("NEON x2:    %10.2f ms\n", 1000*bench2());
        printf("NEON ARRAY: %10.2f ms\n", 1000*bench_array());
#endif
    } else {
        int i;
        uint64_t init[4]={
            UINT64_C(0x12345), 
            UINT64_C(0x23456), 
            UINT64_C(0x34567), 
            UINT64_C(0x45678)
        }, length=4;

        init_by_array64(init, length);
        printf("1000 outputs of genrand64_int64()\n");
        
        for (i=0; i<1000; i++) {
            printf("%20llu ", genrand64_int64());
            if (i%5==4) printf("\n");
        }
        printf("\n1000 outputs of genrand64_real2()\n");
        for (i=0; i<1000; i++) {
            printf("%10.8f ", genrand64_real2());
            if (i%5==4) printf("\n");
        }
    }
}
