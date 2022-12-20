#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <random>
#include <fstream>

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

global_variable u32 used = 0;
struct memory_arena
{
    size_t Used;
    size_t Size;
    u8 *Base;
};
global_variable memory_arena MemoryArena = {};

#define PushStruct(Arena, Type) (Type *)PushSize_(Arena, sizeof(Type))
inline f32 *PushSize_(memory_arena *Arena, size_t SizeToReserve)
{   
    printf("Want to reserve %d, free %d\n", SizeToReserve, Arena->Size - (Arena->Used + SizeToReserve));
    assert(Arena->Used + SizeToReserve <= Arena->Size);
    void *Result = Arena->Base + Arena->Used;
    Arena->Used += SizeToReserve;
    return (f32*)Result;
}

int reverseInt (u32 val) 
{
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}
f32 dataset[500][28*28];
void read_digit(const char *filename, int idx)
{
    std::ifstream file (filename);
    if (file.is_open())
    {  
        for(int i=0;i<100;++i)
        {
            for(int r=0;r<28;++r)
            {
                for(int c=0;c<28;++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    dataset[(idx+i)][r*28+c] = (temp-127.0f)/255.0f;
                }
            }
        }
    } else
    {
        printf("CANT OPEN\n");
    }

}

f32 Sigmoid(f32 x)
{
    f32 r = 1 / (1 + exp(-x));
    return r;
}

f32 D_Sigmoid(f32 X)
{
    return X * (1.0f- X);
}
// ReLU activation function
double Relu(double x){
    return fmax(0.0,x);
}


// Derivative of ReLU activation function
double D_Relu(double x){
    if(x < 0.0){
        return 0.0;
    }
    return 1.0;
}


f32 Square(f32 X)
{
    return X * X;
}

f32 get_next_random()
{
    return (float(rand())/RAND_MAX) - 0.5;
}


void init_weights(f32* w, u32 in, u32 out, f32 v = 1)
{
    for(u32 j=0;j<out;++j)
    {
        for(u32 i=0;i<in;++i)
        {
            w[j * in + i] = v == 1?get_next_random()*0.1:0.0f;
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
void shuffle(u32 *array, size_t n)
{
    if (n > 1) 
    {
        size_t i;
        for (i = 0; i < n - 1; i++) 
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          u32 t = array[j];
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
        
        if(activation) {
            out_data[i] = activation(out_data[i]);
        }
        //printf("%f, ", out_data[i]);
    }
    
}
void calc_error(f32* output, f32* target,f32* out_errors, u32 out_size, f32 (*d_activation)(f32) = NULL)
{
    for(u32 k=0;k<out_size;++k)
    {
        f32 d = d_activation != NULL ? d_activation(output[k]) : 1.0f;
        out_errors[k] = (output[k]-target[k]) * d;
    }
}

void backward(f32* in_dl, f32* out_dl, f32* w, f32* a, u32 in_size, u32 out_size, f32 (*d_activation)(f32))
{
    for(u32 k=0;k<in_size;++k)
    {
        f32 accum = 0;
        for (u32 l=0;l<out_size;++l)
        {
            //accum += in_dl[l] * w[k*out_size+l];
            accum += in_dl[l] * w[l*in_size+k];
        }
        //  a[k] * (1-a[k])
        f32 d = d_activation ? d_activation(a[k]) : 1.0f;
        // a[k] > 0.0 ? 1.0f : 0.1f;
        out_dl[k] = accum * d;
    }
}

#define PI 3.141549

void normalize(f32*d, u8* pixels)
{
    for(u32 i=0;i<784;++i)
    {
        d[i] = (pixels[i] - 127.0f) / 255.0f;
    }
}

u32* get_indices(i32 size)
{
    u32* indices = (u32*)malloc(sizeof(u32) * size);
    for(u32 i=0;i<size;++i){
        indices[i] = rand() % size;
    }
    return indices;
}

void print_m(f32* w, u32 rows, u32 cols)
{
    for(u32 j=0;j<rows;++j)
    {
        for(u32 i=0;i<cols;++i)
        {
            printf("%f ", w[j * cols + i]);
        }
        printf("\n");
    }
}

void decToBinary(f32* out, int n)
{
    // array to store binary number
    int binaryNum[32] = {};

    // counter for binary array
    int i = 0;
    while (n > 0) {

        // storing remainder in binary array
        binaryNum[i] = n % 2;
        n = n / 2;
        i++;
    }

    // printing binary array in reverse order
    for (int j = 9; j >= 0; j--)
        out[j] = binaryNum[j];
}

f32 softmax(float* array, f32 x, int size) {
    // find the maximum value in the array
    float max = array[0];
    for (int i = 1; i < size; i++) {
        if (array[i] > max) {
            max = array[i];
        }
    }

    // subtract the maximum value from each element in the array
    for (int i = 0; i < size; i++) {
        array[i] -= max;
    }

    // calculate the sum of the exponentials of the array elements
    float sum = 0;
    for (int i = 0; i < size; i++) {
        sum += exp(array[i]);
    }

    // divide each element in the array by the sum of the exponentials
    for (int i = 0; i < size; i++) {
        array[i] = exp(array[i]) / sum;
    }

    return x;
}

struct layer{
    f32* w;
    f32* b;
    f32* a;
    u16 in_size;
    u16 out_size;
    f32 (*activation)(f32);
    f32 (*d_activation)(f32);
    f32* dl;
    f32* dw;
    f32* vdw;
    f32* vdb;
};

struct neural_n
{
    layer layers[10];
    u32 size;
};

void print_layer_output(neural_n* n, u32 l)
{
    printf("a:\n");
    for(u32 i=0;i<n->layers[l].out_size;++i)
        printf("%f, ", n->layers[l].a[i]);
    
    printf("\n");
}

void print_layer_w(neural_n* n, u32 l)
{
    u32 out_size = n->layers[l].out_size;
    u32 in_size = n->layers[l].in_size;
    for(u32 i=0;i<out_size;++i){
        for(u32 j=0;j<in_size;++j){
            printf("%f, ", n->layers[l].w[i*in_size + j]);
        }
        printf("\n");
    }
    printf("--------\n");
}

void print_layer_error(neural_n* n, u32 l)
{
    printf("Errors:\n");
    for(u32 i=0;i<n->layers[l].out_size;++i)
        printf("%f, ", n->layers[l].dl[i]);
    
    printf("\n");
    
}

void print_layer_b(neural_n* n, u32 l)
{
    printf("Bias:\n");
    for(u32 i=0;i<n->layers[l].out_size;++i)
        printf("%f, ", n->layers[l].b[i]);
    
    printf("\n");
}

void print_info(neural_n*n)
{
    for(u32 i=0;i<n->size;++i){
        printf("--------------------------\n");
        printf("Layer %d\nWeights:\n", i);
        print_layer_w(n,i);
        print_layer_b(n,i);
        print_layer_output(n,i);
        print_layer_error(n,i);
        printf("--------------------------\n");
    }
}


void add_layer(neural_n* n, u32 in_size, u32 out_size, f32 (*activation)(f32) = NULL, f32 (*d_activation)(f32) = NULL)
{
    layer* layer = n->layers + n->size;
    layer->in_size = in_size;
    layer->out_size = out_size;
    layer->w = PushSize_(&MemoryArena, out_size*in_size*sizeof(f32));
    layer->b = PushSize_(&MemoryArena, out_size*sizeof(f32));
    layer->a = PushSize_(&MemoryArena, out_size*sizeof(f32));
    layer->dl = PushSize_(&MemoryArena, out_size*sizeof(f32));
    layer->vdw = PushSize_(&MemoryArena, out_size*in_size*sizeof(f32));
    layer->vdb = PushSize_(&MemoryArena, out_size*sizeof(f32));

    init_weights(layer->w, in_size, out_size);
    init_weights(layer->b,out_size,1);
    init_weights(layer->vdw, in_size, out_size, 0);
    init_weights(layer->vdb, out_size, 1,0);
    layer->activation = activation;
    layer->d_activation = d_activation;
 
    n->size += 1;
}

f32* forward(neural_n* n, f32* input){    
    assert(n->size >= 1);
    forward(input, n->layers[0].a, n->layers[0].w, n->layers[0].b, n->layers[0].in_size, n->layers[0].out_size, n->layers[0].activation);
    for(u32 i=1;i<n->size;++i) {
        forward(n->layers[i-1].a, n->layers[i].a, n->layers[i].w, n->layers[i].b, n->layers[i].in_size, n->layers[i].out_size, n->layers[i].activation);
    }
    return n->layers[n->size-1].a;
}

f32 cross_entropy(f32* target, f32* predicted, u32 size)
{
    f32 e = 0.0f;
    for(u32 i=0;i<size;++i){
        e += -target[i] * log(predicted[i]) - (1 - target[i]) * log(1 - predicted[i]);
    }
    return e;
}

f32 sme(f32* target, f32* predicted, u32 size)
{
    
    f32 e = 0.0f;
    for(u32 i=0;i<size;++i){
        // printf("%f, %f\n", target[i], predicted[i]);
        e += Square(target[i] - predicted[i]);    
    }
    return e;
}

f32 backward(neural_n* n, f32* input, f32* target, f32 (*loss)(f32*, f32*, u32), void (*opt)(neural_n*, f32*, f32), f32 lr=0.01)
{
    assert(n->size >= 1);
    forward(input, n->layers[0].a, n->layers[0].w, n->layers[0].b, n->layers[0].in_size, n->layers[0].out_size, n->layers[0].activation);

    for(u32 i=1;i<n->size;++i) {
        forward(n->layers[i-1].a, n->layers[i].a, n->layers[i].w, n->layers[i].b, n->layers[i].in_size, n->layers[i].out_size, n->layers[i].activation);
    }

    calc_error(n->layers[n->size-1].a, target, n->layers[n->size-1].dl, n->layers[n->size-1].out_size, n->layers[n->size-1].d_activation);

    for(u32 i=n->size-1;i>0;--i) {
        backward(n->layers[i].dl, n->layers[i-1].dl, n->layers[i].w, n->layers[i-1].a, n->layers[i].in_size, n->layers[i].out_size, n->layers[i-1].d_activation);
    }

    opt(n, input, lr);

    return loss(target, n->layers[n->size-1].a, n->layers[n->size-1].out_size);
}

void sgd(neural_n* n, f32* input, f32 lr)
{
    // UPDATE WEIGHTS
    for(u32 i=1;i<n->size;++i){   
        for(u32 k=0;k<n->layers[i].out_size;++k){
            for(u32 l=0;l<n->layers[i].in_size;++l)  {
                n->layers[i].w[k * n->layers[i].out_size + l] -= lr * n->layers[i].dl[k] * n->layers[i-1].a[l];
            }
            n->layers[i].b[k] -= lr * n->layers[i].dl[k];
        }
    }

    for(u32 k=0;k<n->layers[0].out_size;++k){
        for(u32 l=0;l<n->layers[0].in_size;++l)
        {
            n->layers[0].w[k * n->layers[0].out_size + l] -= lr * n->layers[0].dl[k] * input[l];
        }

        n->layers[0].b[k] -= lr * n->layers[0].dl[k];
    }
}

void rmsprop(neural_n* n, f32* input, f32 lr)
{
    f32 beta = 0.9;
    for(u32 i=1;i<n->size;++i){   
        for(u32 k=0;k<n->layers[i].out_size;++k){
            for(u32 l=0;l<n->layers[i].in_size;++l)  {
                f32 dw = n->layers[i].dl[k] * n->layers[i-1].a[l];
                f32 sdw = beta * n->layers[i].vdw[k * n->layers[i].in_size + l] + (1.0f -beta)*dw*dw;
                n->layers[i].vdw[k * n->layers[i].in_size + l]  = sdw;
                n->layers[i].w[k * n->layers[i].in_size + l] -= lr * dw / sqrt(sdw);
            }
            f32 db = n->layers[i].dl[k];
            f32 sdb = beta * n->layers[i].vdb[k] + (1.0f-beta)*db*db;
            n->layers[i].vdb[k] = sdb;
            n->layers[i].b[k] -= lr * db / sqrt(sdb);
        }
    }

    for(u32 k=0;k<n->layers[0].out_size;++k){
        for(u32 l=0;l<n->layers[0].in_size;++l)
        { 
            f32 dw = n->layers[0].dl[k] * input[l];
            f32 sdw = beta * n->layers[0].vdw[k * n->layers[0].in_size + l] + (1.0f -beta)*dw*dw;
            n->layers[0].vdw[k * n->layers[0].in_size + l]  = sdw;
            n->layers[0].w[k * n->layers[0].in_size + l] -= lr * dw / sqrt(sdw);
        }
            f32 db = n->layers[0].dl[k];
            f32 sdb = beta * n->layers[0].vdb[k] + (1.0f-beta)*db*db;
            n->layers[0].vdb[k] = sdb;
            n->layers[0].b[k] -= lr * db / sqrt(sdb);
    }
}

void print_to_python(f32* input, int x)
{
    printf("x = [");
    for (u32 i=0;i<x;++i) 
    {
        printf("%f,", input[i]);
    }
    printf("]");
    printf("\n");
}

f32* one_hot(f32 i, f32 num_classes)
{
    f32* target = PushSize_(&MemoryArena, num_classes);
    for(u32 k=0;k<num_classes;++k) {
        target[k] = i == k ? 1.0f : 0.0f;
    }
    return target;
}

f32* gen_n_random(u32 size)
{
    f32* output = PushSize_(&MemoryArena, size);
    for(u32 i=0;i<size;++i) output[i] = ((double)rand()/(double)RAND_MAX) - 0.5f;
    
    return output;
}


#define OLD 0   
int main(int argc, char** argv)
{
    read_digit("data0",0);
    read_digit("data1",100);
    read_digit("data2",200);
    read_digit("data3",300);
    read_digit("data4",400);
 
    srand(time(NULL));
    MemoryArena.Base = (u8 *)malloc(1024*31);
    memset(MemoryArena.Base, 0, 1024*31);
    MemoryArena.Size = 1024*31;
    MemoryArena.Used = 0;
    
    neural_n n = {};
    add_layer(&n, 784, 5, Sigmoid, D_Sigmoid);
    u32 Used = MemoryArena.Used;
    printf("USED MEMORY %d", Used);
    f32* target = PushSize_(&MemoryArena, 5 * sizeof(f32));
    u32* indices = get_indices(500);
    u32 epochs = 2000;
    f32 lr = 0.001;
    for(u32 i=0;i<epochs;++i){
        shuffle(indices, 500);
        f32 error = 0;
        for(u32 p=0;p<500;++p){
            MemoryArena.Used = Used;
            f32* in = dataset[indices[p]];
            for(u32 k=0;k<5;++k) target[k] = indices[p]/100 == k ? 1.0f : 0.0f;
            f32 errors = backward(&n, in, target, sme, rmsprop, lr);
            error += errors;
        }

        if(i%100==0)printf("error [%d/%d] %f %f\n", i, epochs, error, error / 500.0f);
    }

    // f32* ins = gen_n_random(600);
    // printf("x = [");
    // for (u32 i=0;i<600;++i) 
    // {
    //     printf("%f,", ins[i] * PI * 2.0f);
    // }
    // printf("]");
    // printf("\n");
    // printf("y = [");
    // for (u32 i=0;i<600;++i) 
    // {
    //     f32* output = forward(&n, &ins[i]);
    //     printf("%f,", output[0]);
    // }
    // printf("]\n");

    // while(1){
    // int myInt;
    // printf("\nINTRODUCE NUMBER\n");
    // scanf("%d", &myInt);
    // f32* inputs = dataset[myInt*100+1];
    // for(u32 i=0;i<28;++i)
    // {
    //     for(u32 j=0;j<28;++j){
    //         if(inputs[i*28+j] > 0.0) printf("#");
    //         else printf(".");
    //     }
    //     printf("\n");
    // }
    // f32* outputs = forward(&n, inputs);
    // int max = 0;
    // int max_i = 0;
    // for (u32 i=0;i<5;++i) {
    //     if( outputs[i] > max ){
    //         max = outputs[i];
    //         max_i = i;
    //     }
    //     printf("%f, ", outputs[i]); 
    // }
    // printf("\nNN predicted: %d", max_i); 
    // }
}
