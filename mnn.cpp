#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

typedef float f32;
typedef double f64;

typedef uint32_t bool32;

#define global_variable static
#define local_persist static
#define internal static

#define NUM_E 2.71828
global_variable u32 used = 0;
struct memory_arena
{
    size_t Used;
    size_t Size;
    f32 *Base;
};
global_variable memory_arena MemoryArena = {};

#define PushStruct(Arena, Type) (Type *)PushSize_(Arena, sizeof(Type))
inline f32 *PushSize_(memory_arena *Arena, size_t SizeToReserve)
{   
    assert(Arena->Used + (SizeToReserve) <= Arena->Size);
    f32 *Result = Arena->Base + (Arena->Used);
    Arena->Used += SizeToReserve;
    return Result;
}

#include "matrix.cpp"
f32 Sigmoid(f32 X)
{
    return 1.0f / (f32)(1.0f + exp(-X));
}

matrix Sigmoid(matrix Z)
{
    return MapStatic(Z, Sigmoid);
}

f32 Square(f32 X)
{
    return X * X;
}

matrix Square(matrix Z)
{
    return MapStatic(Z, Square);
}

f32 get_next_random()
{
    float Rando = float(rand())/RAND_MAX;
    return Rando;
}

#define INPUT_LAYER_SIZE 1
#define HIDDEN_LAYER_SIZE 32
#define HIDDEN_LAYER2_SIZE 32
#define OUTPUT_LAYER_SIZE 1


matrix init_weights(u32 in, u32 out)
{
    matrix r = Matrix(out, in);
    for(u32 j=0;j<out;++j)
    {
        for(u32 i=0;i<in;++i)
        {
            Set(&r, j,i, get_next_random());
        }
    }
    return r;
} 

void shuffle(int *array, size_t n)
{
    if (n > 1) 
    {
        size_t i;
        for (i = 0; i < n - 1; i++) 
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          int t = array[j];
          array[j] = array[i];
          array[i] = t;
        }
    }
}

void shuffle(f32 *array, size_t n)
{
    if (n > 1) 
    {
        size_t i;
        for (i = 0; i < n - 1; i++) 
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          f32 t = array[j];
          array[j] = array[i];
          array[i] = t;
        }
    }
}

struct layer
{
    u16 input;
    u16 output;
    matrix w;
    matrix b;
    bool32 last;
};

layer Layer(u16 input, u16 output, bool32 last = false)
{
    layer Result = {};
    Result.w = init_weights(input, output);
    Result.b = init_weights(1, output);
    Result.last = last;
    Result.input = input;
    Result.output = output;
    return Result;
}

matrix Forward(matrix Input, layer Layer)
{
    return Sigmoid((Layer.w * Input) + Layer.b);
}


#define PI 3.141549

int main(int argc, char** argv)
{
    srand(time(NULL));
    f32 lr = 0.1 ;
    MemoryArena.Base = (f32 *)malloc(64*1024*sizeof(u8));
    MemoryArena.Size = 64*1024 * sizeof(u8);
    MemoryArena.Used = 0;

    layer InputLayer = Layer(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE);
    layer HiddenLayer = Layer(HIDDEN_LAYER_SIZE, HIDDEN_LAYER2_SIZE);
    layer HiddenLayer2 = Layer(HIDDEN_LAYER2_SIZE, OUTPUT_LAYER_SIZE, true);
    f32* indices = PushSize_(&MemoryArena, 600*sizeof(f32));
    for (u32 i=0;i<600;++i) 
    {
        indices[i] = ((double)rand()/(double)RAND_MAX) * 2 *PI;
    }
    u32 epochs = 1000;
    matrix input = Matrix(1,1);
    matrix output = Matrix(1,1);

    u32 Used = MemoryArena.Used;

    matrix h = Matrix(HIDDEN_LAYER_SIZE, 1);
    matrix h1 = Matrix(HIDDEN_LAYER2_SIZE, 1);
    matrix o = Matrix(OUTPUT_LAYER_SIZE, 1);
    printf("Memory needed %d:\n", Used);
    u32 max = 0;
    for(u32 i=0;i<epochs;++i){
        shuffle(indices,600);
        f32 error = 0;
        for(u32 j=0;j<600;++j)
        {
            MemoryArena.Used = Used;

            // forward propagation
            Set(&input, 0,0, indices[j]);
            Set(&output, 0,0, sin(indices[j]));
            h = Forward(input, InputLayer);
            h1 = Forward(h, HiddenLayer);
            o = Forward(h1, HiddenLayer2);

            // backpropagation
            matrix dlo = Hadamard(lr,o,1-o,o-output);
            matrix dldw = Hadamard(lr, h1,1-h1, Transpose(HiddenLayer2.w) * dlo);
            matrix dldi = Hadamard(lr, h,1-h, Transpose(HiddenLayer.w) * dldw);

            HiddenLayer2.w -=  dlo * Transpose(h1);
            HiddenLayer.w -=  dldw * Transpose(h);
            InputLayer.w -=  dldi * Transpose(input);

            HiddenLayer2.b -= dlo;
            HiddenLayer.b -= dldw;
            InputLayer.b -= dldi;

            matrix l = Square(output - o);
            error+= l.Data[0];
            max = MemoryArena.Used > max ? MemoryArena.Used : max;
        }
        if(i%50==0) printf("error [%d/%d] %f \n", i, epochs, error/600.0f);
    }
    MemoryArena.Used = Used;

    f32 results[600];
    printf("MAXIMO %d %zd", max, MemoryArena.Used);
    // forward propagation
    for (u32 i=0;i<600;++i) 
    {
        indices[i] = ((double)rand()/(double)RAND_MAX) * 2*PI;
        Set(&input, 0,0, indices[i]);
        h = Forward(input, InputLayer);
        h1 = Forward(h, HiddenLayer);
        o = Forward(h1, HiddenLayer2);
        results[i] = o.Data[0];
        MemoryArena.Used = Used;

    }

    for (u32 i=0;i<600;++i) 
    {
        printf("%f,", indices[i]);
    }
    printf("--------------------------------------------------");
    for (u32 i=0;i<600;++i) 
    {
        printf("%f,", results[i]);
    }

}
