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

f32 DSigmoid(f32 X)
{
    return Sigmoid(X) * (1.0f - Sigmoid(X));
}

f32 get_next_random()
{
    float Rando = float(rand())/RAND_MAX;
    return Rando;
}

#define INPUT_LAYER_SIZE 1
#define HIDDEN_LAYER_SIZE 16
#define HIDDEN_LAYER2_SIZE 16
#define OUTPUT_LAYER_SIZE 1


matrix weight12;
matrix weight23;
matrix weight45;

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

#define PI 3.141549

int main(int argc, char** argv)
{

    srand(time(NULL));
    f32 lr = 0.1;

    weight12 = init_weights(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE);
    weight23 = init_weights(HIDDEN_LAYER_SIZE, HIDDEN_LAYER2_SIZE);
    weight45 = init_weights(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE);

    matrix b1 = init_weights(1,HIDDEN_LAYER_SIZE);
    matrix b2 = init_weights(1,HIDDEN_LAYER2_SIZE);
    matrix b3 = init_weights(1,OUTPUT_LAYER_SIZE);
    f32 indices[600];
    for (u32 i=0;i<600;++i) 
    {
        indices[i] = ((double)rand()/(double)RAND_MAX) * PI;
    }
    u32 epochs = 2000;
    for(u32 i=0;i<epochs;++i){
        shuffle(indices,600);
        f32 error = 0;
        for(u32 j=0;j<600;++j)
        {
            // forward propagation
            matrix input = Matrix(1,1);
            Set(&input, 0,0, indices[j]);
            matrix output = Matrix(1,1);
            Set(&output, 0,0, sin(indices[j]));
            matrix h = Sigmoid((weight12 * input) + b1);
            matrix h1 = Sigmoid((weight23 * h) + b2);
            matrix o = Sigmoid((weight45 * h1) + b3);

            // cost
            matrix l = Square(output - o);
            // backpropagation
            matrix dlo = Hadamard(Hadamard(o,1-o),o-output);
            matrix dldw = Hadamard(Hadamard(h1,1-h1), Transpose(weight45) * dlo);
            matrix dldi = Hadamard(Hadamard(h,1-h), Transpose(weight23) * dldw);
            // matrix dldw = Hadamard(Hadamard(h,1-h), Transpose(weight23) * dlo) * Transpose(input[indices[j]]);
            weight45 = weight45 - lr * dlo * Transpose(h1);
            weight23 = weight23 - lr * dldw * Transpose(h);
            weight12 = weight12 - lr * dldi * Transpose(input);
            b1 = b1 - lr * dldi;
            b2 = b2 - lr * dldw;
            b3 = b3 - lr * dlo;

            l = Square(output - o);
            error+= l.Data[0];
        }
        printf("error [%d/%d] %f \n", i, epochs, error/600.0f);
    }
    // forward propagation
    matrix input = Matrix(1,1);
    Set(&input, 0,0, 1.6);
    matrix h = Sigmoid((weight12 * input) + b1);
    matrix h1 = Sigmoid((weight23 * h) + b2);
    matrix o = Sigmoid((weight45 * h1) + b3);
    printf("----\n");
    Print(o);
    printf("----\n");

    Set(&input, 0,0, 0);
    h = Sigmoid((weight12 * input) + b1);
    h1 = Sigmoid((weight23 * h) + b2);
    o = Sigmoid((weight45 * h1) + b3);
    printf("----\n");
    Print(o);
    printf("----\n");

}
