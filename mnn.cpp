#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <random>

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


f32 Square(f32 X)
{
    return X * X;
}

std::random_device rd;


std::mt19937 e2(rd());

std::uniform_real_distribution<f32> dist(-1, 1);
f32 get_next_random()
{
    return dist(e2);
    //return (float(rand())/RAND_MAX) - 0.5;
}

#define INPUT_LAYER_SIZE 1
#define HIDDEN_LAYER_SIZE 16
#define HIDDEN_LAYER2_SIZE 16
#define OUTPUT_LAYER_SIZE 1

void init_weights(f32* w, u32 rows, u32 cols)
{
    for(u32 j=0;j<rows;++j)
    {
        for(u32 i=0;i<cols;++i)
        {
            w[j * cols + i] = get_next_random()*0.1;
        }
    }
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

void forward(f32* in_data, f32* out_data, f32* w, f32* bias, u32 in_size, u32 out_size, f32 (*activation)(f32)=NULL)
{
    for(u32 i=0;i<out_size;++i)
    {
        f32 accum = 0;
        for(u32 j=0;j<in_size;++j)
        {
            accum += in_data[j] * w[i*in_size + j];
        }
        out_data[i] = accum + bias[i];
        if(activation) out_data[i] = activation(out_data[i]);
    }
}

void calc_error(f32* output, f32* target,f32* out_errors, u32 out_size)
{
    for(u32 k=0;k<out_size;++k)
    {
        out_errors[k] = (output[k]-target[k]); //* output[k] * (1-output[k]);
    }
}
// backward(output_errors, hidden2_errors, HiddenLayer2.w.Data, Hidden1, HIDDEN_LAYER2_SIZE, OUTPUT_LAYER_SIZE);
// backward(hidden2_errors, hidden_errors, w2, Hidden, HIDDEN_LAYER2_SIZE, HIDDEN_LAYER_SIZE);
void backward(f32* in_dl, f32* out_dl, f32* w, f32* a, u32 in_size, u32 out_size)
{
    for(u32 k=0;k<in_size;++k)
    {
        f32 accum = 0;
        for (u32 l=0;l<out_size;++l)
        {
            //accum += in_dl[l] * w[l*out_size+k];
            accum += in_dl[l] * w[l*in_size+k];
        }
        out_dl[k] = accum * a[k] * (1-a[k]);
    }
}

#define PI 3.141549

int main(int argc, char** argv)
{
    srand(time(NULL));
    f32 lr = 0.01;
    MemoryArena.Base = (f32 *)malloc(1024*sizeof(f32));
    MemoryArena.Size = 1024 * sizeof(f32);
    MemoryArena.Used = 0;

    f32* w1 = PushSize_(&MemoryArena, INPUT_LAYER_SIZE * HIDDEN_LAYER_SIZE);
    f32* w2 = PushSize_(&MemoryArena, HIDDEN_LAYER_SIZE * HIDDEN_LAYER2_SIZE);
    f32* w3 = PushSize_(&MemoryArena, HIDDEN_LAYER2_SIZE * OUTPUT_LAYER_SIZE);

    f32* b1 = PushSize_(&MemoryArena, HIDDEN_LAYER_SIZE);
    f32* b2 = PushSize_(&MemoryArena, HIDDEN_LAYER2_SIZE);
    f32* b3 = PushSize_(&MemoryArena, OUTPUT_LAYER_SIZE);

    init_weights(b1, 1, HIDDEN_LAYER_SIZE);
    init_weights(b2, 1, HIDDEN_LAYER2_SIZE);
    init_weights(b3, 1, OUTPUT_LAYER_SIZE);

    init_weights(w1, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE);
    init_weights(w2, HIDDEN_LAYER_SIZE, HIDDEN_LAYER2_SIZE);
    init_weights(w3, HIDDEN_LAYER2_SIZE, OUTPUT_LAYER_SIZE);

    u32 epochs = 20000;
    matrix input = Matrix(1,1);
    matrix output = Matrix(1,1);

    f32* Hidden = PushSize_(&MemoryArena, HIDDEN_LAYER_SIZE) ;
    f32* Hidden1 = PushSize_(&MemoryArena, HIDDEN_LAYER2_SIZE) ;
    f32* Output = PushSize_(&MemoryArena, OUTPUT_LAYER_SIZE) ;

    f32* output_errors = PushSize_(&MemoryArena, OUTPUT_LAYER_SIZE);
    f32* hidden2_errors = PushSize_(&MemoryArena, HIDDEN_LAYER2_SIZE);
    f32* hidden_errors = PushSize_(&MemoryArena, HIDDEN_LAYER_SIZE);
    
    f32* indices = PushSize_(&MemoryArena, 600);
    for (u32 i=0;i<600;++i)  {
        indices[i] = ((double)rand()/(double)RAND_MAX) * PI * 2 ;
    }
    u32 Used = MemoryArena.Used;

    //printf("Memory needed %d:\n", Used);
    u32 max = 0;
    for(u32 i=0;i<epochs;++i){
        shuffle(indices,600);
        f32 error = 0;
        if(i%1000==0) lr *= 0.7;
        for(u32 j=0;j<600;++j)
        {
            MemoryArena.Used = Used;

            // forward propagation
            Set(&input, 0,0, indices[j]);
            Set(&output, 0,0, sin(indices[j]));
            forward(input.Data, Hidden, w1, b1, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, Sigmoid);
            forward(Hidden, Hidden1, w2, b2, HIDDEN_LAYER_SIZE, HIDDEN_LAYER2_SIZE, Sigmoid);
            forward(Hidden1, Output, w3, b3, HIDDEN_LAYER2_SIZE, OUTPUT_LAYER_SIZE);

            // error for output layer
            calc_error(Output, output.Data, output_errors, OUTPUT_LAYER_SIZE);
            // Calculate error delta
            backward(output_errors, hidden2_errors, w3, Hidden1, HIDDEN_LAYER2_SIZE, OUTPUT_LAYER_SIZE);
            backward(hidden2_errors, hidden_errors, w2, Hidden, HIDDEN_LAYER2_SIZE, HIDDEN_LAYER_SIZE);

            // UPDATE WEIGHTS
            for(u32 k=0;k<OUTPUT_LAYER_SIZE;++k)
            {
                for(u32 l=0;l<HIDDEN_LAYER2_SIZE;++l)
                {
                    w3[k * OUTPUT_LAYER_SIZE + l] -= lr * output_errors[k] * Hidden1[l];
                }
                b3[k] -= lr * output_errors[k];

            }

            for(u32 k=0;k<HIDDEN_LAYER2_SIZE;++k)
            {
                for(u32 l=0;l<HIDDEN_LAYER_SIZE;++l)
                {
                    w2[k * HIDDEN_LAYER2_SIZE + l] -= lr * hidden2_errors[k] * Hidden[l];
                }
                b2[k] -= lr * hidden2_errors[k];

            }

            for(u32 k=0;k<HIDDEN_LAYER_SIZE;++k)
            {
                for(u32 l=0;l<INPUT_LAYER_SIZE;++l)
                {
                    w1[k * HIDDEN_LAYER_SIZE + l] -= lr * hidden_errors[k] * input.Data[l];
                }

                b1[k] -= lr * hidden_errors[k];
            }

            error += Square(output.Data[0] - Output[0]);
            max = MemoryArena.Used > max ? MemoryArena.Used : max;
        }
        if(i%50==0) printf("error [%d/%d] %f \n", i, epochs, error/600.0f);
    }
    MemoryArena.Used = Used;
    f32 inputs[600];
    f32 outputs[600];
    for(u32 j=0;j<600;++j)
    {
        inputs[j] = ((double)rand()/(double)RAND_MAX)  * PI * 2 ;
        forward(&inputs[j], Hidden, w1, b1,INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, Sigmoid);
        forward(Hidden, Hidden1, w2, b2,HIDDEN_LAYER_SIZE, HIDDEN_LAYER2_SIZE, Sigmoid);
        forward(Hidden1, Output, w3, b3,HIDDEN_LAYER2_SIZE, OUTPUT_LAYER_SIZE);
        outputs[j] = Output[0];
    }

    // printf("x = [");
    // for (u32 i=0;i<600;++i) 
    // {
    //     printf("%f,", inputs[i]);
    // }
    // printf("]");
    // printf("\n");
    // printf("y= [");
    // for (u32 i=0;i<600;++i) 
    // {
    //     printf("%f,", outputs[i]);
    // }
    // printf("]");

}
