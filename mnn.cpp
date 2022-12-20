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

typedef double f32;
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

int reverseInt (u32 val) 
{
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

f32 dataset[100*10][28*28];
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
                    dataset[idx+i][r*28+c] = (temp-127.0f)/255.0f;
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

f32 Relu(f32 X)
{
    return X > 0.0f ? X : 0.1f * X;
}

f32 D_Sigmoid(f32 X)
{
    return X * (1.0f- X);
}

f32 D_Relu(f32 X)
{
    return X > 0.0f ? 1.0f : 0.1f;
}


f32 Square(f32 X)
{
    return X * X;
}

f32 get_next_random()
{
    return (float(rand())/RAND_MAX) - 0.5;
}

#define INPUT_LAYER_SIZE 784
#define HIDDEN_LAYER_SIZE 6
#define HIDDEN_LAYER2_SIZE 5
#define OUTPUT_LAYER_SIZE 5

void init_weights(f32* w, u32 in, u32 out)
{
    // for(u32 j=0;j<rows;++j)
    // {
    //     for(u32 i=0;i<cols;++i)
    //     {
    //         w[j * cols + i] = get_next_random();
    //     }
    // }

    for(u32 j=0;j<out;++j)
    {
        for(u32 i=0;i<in;++i)
        {
            w[j * in + i] = get_next_random() * 0.01f;
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

void get_indices(u32* out, i32 indices)
{
    for(u32 i=0;i<indices;++i){
        out[i] = rand() % 10000;
    }
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
    u32 in_size;
    u32 out_size;
    f32 (*activation)(f32);
    f32 (*d_activation)(f32);
    f32* dl;
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
    layer->w = PushSize_(&MemoryArena, out_size*in_size);
    layer->b = PushSize_(&MemoryArena, out_size);
    layer->a = PushSize_(&MemoryArena, out_size);
    layer->dl = PushSize_(&MemoryArena, out_size);

    init_weights(layer->w, in_size, out_size);
    init_weights(layer->b, out_size, 1);
    layer->activation = activation;
    layer->d_activation = d_activation;
    // printf("---------------------------\n");
    // printf("added layer at %d with %d inputs and %d outputs\n", n->size,layer->in_size, layer->out_size);
    // printf("Weights:\n");
    // print_layer_w(n,n->size);
    // print_layer_b(n,n->size);
    // print_layer_output(n,n->size);
    // print_layer_error(n,n->size);

    // printf("--------------------------\n");
    n->size += 1;
}

f32* forward(neural_n* n, f32* input){

    forward(input,          n->layers[0].a, n->layers[0].w, n->layers[0].b, n->layers[0].in_size, n->layers[0].out_size, n->layers[0].activation);
    forward(n->layers[0].a, n->layers[1].a, n->layers[1].w, n->layers[1].b, n->layers[1].in_size, n->layers[1].out_size, n->layers[1].activation);
    forward(n->layers[1].a, n->layers[2].a, n->layers[2].w, n->layers[2].b, n->layers[2].in_size, n->layers[2].out_size, n->layers[2].activation);
    return n->layers[2].a;
}

f32 backward(neural_n* n, f32* input, f32* target, f32 lr = 0.001)
{
    forward(input,          n->layers[0].a, n->layers[0].w, n->layers[0].b, n->layers[0].in_size, n->layers[0].out_size, n->layers[0].activation);
    forward(n->layers[0].a, n->layers[1].a, n->layers[1].w, n->layers[1].b, n->layers[1].in_size, n->layers[1].out_size, n->layers[1].activation);
    forward(n->layers[1].a, n->layers[2].a, n->layers[2].w, n->layers[2].b, n->layers[2].in_size, n->layers[2].out_size, n->layers[2].activation);

    calc_error(n->layers[2].a, target, n->layers[2].dl, n->layers[2].out_size, n->layers[2].d_activation);
    backward(n->layers[2].dl, n->layers[1].dl, n->layers[2].w, n->layers[1].a, n->layers[2].in_size, n->layers[2].out_size, n->layers[1].d_activation);
    backward(n->layers[1].dl, n->layers[0].dl, n->layers[1].w, n->layers[0].a, n->layers[1].in_size, n->layers[1].out_size, n->layers[0].d_activation);

    // UPDATE WEIGHTS
    for(u32 k=0;k<n->layers[2].out_size;++k)
    {
        for(u32 l=0;l<n->layers[2].in_size;++l)
        {
            n->layers[2].w[k * n->layers[2].out_size + l] -= lr * n->layers[2].dl[k] * n->layers[1].a[l];
        }
        n->layers[2].b[k] -= lr * n->layers[2].dl[k];

    }

    for(u32 k=0;k<n->layers[1].out_size;++k)
    {
        for(u32 l=0;l<n->layers[1].in_size;++l)
        {
            n->layers[1].w[k * n->layers[1].out_size + l] -= lr * n->layers[1].dl[k] * n->layers[0].a[l];
        }
        n->layers[1].b[k] -= lr * n->layers[1].dl[k];

    }

    for(u32 k=0;k<n->layers[0].out_size;++k)
    {
        for(u32 l=0;l<n->layers[0].in_size;++l)
        {
            n->layers[0].w[k * n->layers[0].out_size + l] -= lr * n->layers[0].dl[k] * input[l];
        }

        n->layers[0].b[k] -= lr * n->layers[0].dl[k];
    }
    f32 e = 0;
    for(u32 i=0;i<OUTPUT_LAYER_SIZE;++i){
        //printf("y_true=-%f * log(y_pred=%f)\n", target[i], n->layers[2].a[i]);
        e += -target[i] * log(n->layers[2].a[i]) - (1 - target[i]) * log(1 - n->layers[2].a[i]);
        //printf("e=%f\n",e);
    }
    //getchar();
    //f32 e = Square(target[0] - n->layers[2].a[0]);    
    return e;
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

#define OLD 0   
int main(int argc, char** argv)
{
    read_digit("data0",0);
    read_digit("data1",100);
    read_digit("data2",200);
    read_digit("data3",300);
    read_digit("data4",400);


    srand(time(NULL));
    MemoryArena.Base = (f32 *)malloc(784*1024*sizeof(f32));
    memset(MemoryArena.Base, 0, 784*1024*sizeof(f32));
    MemoryArena.Size = 784*1024 * sizeof(f32);
    MemoryArena.Used = 0;
#if OLD
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
    f32* Hidden = PushSize_(&MemoryArena, HIDDEN_LAYER_SIZE) ;
    f32* Hidden1 = PushSize_(&MemoryArena, HIDDEN_LAYER2_SIZE) ;
    f32* Output = PushSize_(&MemoryArena, OUTPUT_LAYER_SIZE) ;

    f32* output_errors = PushSize_(&MemoryArena, OUTPUT_LAYER_SIZE);
    f32* hidden2_errors = PushSize_(&MemoryArena, HIDDEN_LAYER2_SIZE);
    f32* hidden_errors = PushSize_(&MemoryArena, HIDDEN_LAYER_SIZE);
#endif
    neural_n n = {};
    add_layer(&n, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, Sigmoid, D_Sigmoid);
    add_layer(&n, HIDDEN_LAYER_SIZE, HIDDEN_LAYER2_SIZE, Sigmoid, D_Sigmoid);
    add_layer(&n, HIDDEN_LAYER2_SIZE, OUTPUT_LAYER_SIZE, Sigmoid, D_Sigmoid);

    // print_layer_w(&n, 0);
    // print_layer_output(n, 0);
    // print_layer_w(&n, 1);
    // print_layer_output(n, 1);
    // print_layer_w(&n, 2);
    // print_layer_output(n, 2);

    // f32* indices = PushSize_(&MemoryArena, 600);
    // for (u32 i=0;i<600;++i)  {
    //     indices[i] = ((double)rand()/(double)RAND_MAX);
    // }
    i32 indices[500];
    for (u32 i=0;i<500;++i)  {
        indices[i] = i;
    }

    //print_to_python(indices, 600);
    f32* target = PushSize_(&MemoryArena, 5);

    u32 Used = MemoryArena.Used;
    printf("Memory needed %d:\n", Used);
    u32 epochs = 2000;
    f32 lr = 0.1;

    for(u32 i=0;i<epochs;++i){

        shuffle(indices,500);
        //if(i%100==0) lr *= 0.9;
        f32 error = 0;
        for(u32 p=0;p<500;++p){
      
            MemoryArena.Used = Used;
            f32* input = dataset[indices[p]];
            for(u32 k=0;k<5;++k) target[k] = (indices[p]/100) == k ? 1.0f : 0.0f;
            //target[0] = sin(indices[p] * 2 * PI);


            #if OLD
                forward(input, Hidden, w1, b1, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, Sigmoid);
                forward(Hidden, Hidden1, w2, b2, HIDDEN_LAYER_SIZE, HIDDEN_LAYER2_SIZE, Sigmoid);
                forward(Hidden1, Output, w3, b3, HIDDEN_LAYER2_SIZE, OUTPUT_LAYER_SIZE);

                // error for output layer
                calc_error(Output, target, output_errors, OUTPUT_LAYER_SIZE);
                // Calculate error delta
                backward(output_errors, hidden2_errors, w3, Hidden1, HIDDEN_LAYER2_SIZE, OUTPUT_LAYER_SIZE, D_Sigmoid);
                backward(hidden2_errors, hidden_errors, w2, Hidden, HIDDEN_LAYER_SIZE, HIDDEN_LAYER2_SIZE, D_Sigmoid);

                //UPDATE WEIGHTS
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
                        w1[k * HIDDEN_LAYER_SIZE + l] -= lr * hidden_errors[k] * input[l];
                    }

                    b1[k] -= lr * hidden_errors[k];
                }
                error += Square(target[0] - Output[0]);
            #else
                f32 errors = backward(&n, input, target, lr);
                error += errors;
            #endif

    }
        if(i%10==0)printf("error [%d/%d] %f %f\n", i, epochs, error, error / 600.0f);
    }

    #if OLD
        f32 output[1] = {};
        forward(input, Hidden, w1, b1, INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, Sigmoid);
        forward(Hidden, Hidden1, w2, b2, HIDDEN_LAYER_SIZE, HIDDEN_LAYER2_SIZE, Sigmoid);
        forward(Hidden1, output, w3, b3, HIDDEN_LAYER2_SIZE, OUTPUT_LAYER_SIZE);
    #else
    
    while(1){
    int myInt;
    printf("\nINTRODUCE NUMBER\n");
    scanf("%d", &myInt);
    f32* inputs = dataset[myInt*100];
    for(u32 i=0;i<28;++i)
    {
        for(u32 j=0;j<28;++j){
            if(inputs[i*28+j] > 0.0) printf("#");
            else printf(".");
        }
        printf("\n");
    }
    f32* outputs = forward(&n, inputs);
    int max = 0;
    int max_i = 0;
    for (u32 i=0;i<5;++i) {
        if( outputs[i] > max ){
            max = outputs[i];
            max_i = i;
        }
        printf("%f, ", outputs[i]); 
    }
    printf("\nNN predicted: %d", max_i); 
    }



    #endif
    // printf("x = [");
    // for (u32 i=0;i<600;++i) 
    // {
    //     printf("%f,", inputs[i] * PI * 2.0f);
    // }
    // printf("]");
    // printf("\n");
    // printf("y = [");
    // for (u32 i=0;i<600;++i) 
    // {
    //     printf("%f,", outputs[i]);
    // }
    // printf("]\n");

   

}
