#include <random>

struct matrix
{
    i32 Rows;
    i32 Cols;
    
    f32* Data;
};

f32 GenerateRandomNumber() {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> d{0.5,0.5};
//    return d(gen);
    return ((f32)rand()/(f32)RAND_MAX);;
}

double uniform_distribution(double low, double high) {
	double difference = high - low; // The difference between the two
	int scale = 10000;
	int scaled_difference = (int)(difference * scale);
	return low + (1.0 * (rand() % scaled_difference) / scale);
}


matrix FromArray(f32* Data, i32 Size)
{
    matrix Result = {};

    Result.Cols = 1;
    Result.Rows = Size;
    Result.Data = (f32*) malloc(sizeof(f32) * Size);
    memcpy(Result.Data, Data, sizeof(f32) * Size);

    return Result;
}

matrix Matrix(i32 Rows, i32 Cols)
{
    matrix Result = {};
    Result.Rows = Rows;
    Result.Cols = Cols;

    i32 Size = Rows * Cols;
    if(Size < 4)
    {
	Size = 4;
    }
    Result.Data = (f32*) malloc(sizeof(f32) * Size);
    for(i32 Row = 0; Row < Result.Rows; ++Row)
    {
	for(i32 Col = 0; Col < Result.Cols; ++Col)
	{
	    Result.Data[Row * Result.Cols + Col] = 0;
	}
    }    
    return Result;
}

matrix Clone(matrix A)
{
    matrix Result = Matrix(A.Rows, A.Cols);

    for(i32 Row = 0; Row < Result.Rows; ++Row)
    {
	for(i32 Col = 0; Col < Result.Cols; ++Col)
	{
	    Result.Data[Row * Result.Cols + Col] = A.Data[Row * Result.Cols + Col];
	}
    }    
    return Result;
}

void Map(matrix* A, f32 (*f)(f32))
{
    for(i32 Row = 0; Row < A->Rows; ++Row)
    {
        for(i32 Col = 0; Col < A->Cols; ++Col)
        {
            f32 Nval = f(A->Data[Row * A->Cols + Col]);
            A->Data[Row * A->Cols + Col] = Nval;
        }
    }
}


matrix MapStatic(matrix A, f32 (*f)(f32))
{
    matrix Result = Matrix(A.Rows, A.Cols);
    for(i32 Row = 0; Row < A.Rows; ++Row)
    {
	for(i32 Col = 0; Col < A.Cols; ++Col)
	{
	    f32 Nval = f(A.Data[Row * A.Cols + Col]);
	    Result.Data[Row * Result.Cols + Col] = Nval;
	}
    }

    return Result;
}

matrix
Randomize(i32 Rows, i32 Cols)
{
    int n = 2;
    matrix Result = Matrix(Rows, Cols);
    double min = -1.0 / sqrt(n);
    double max = 1.0 / sqrt(n);
    for(i32 Row = 0; Row < Result.Rows; ++Row)
    {
        for(i32 Col = 0; Col < Result.Cols; ++Col)
        {
            Result.Data[Row * Result.Cols + Col] = uniform_distribution(min, max);
        }
    }

    return Result;
}

void
Print(matrix A)
{
    for(i32 Row = 0; Row < A.Rows; ++Row)
    {
        for(i32 Col = 0; Col < A.Cols; ++Col)
        {
            printf("%f ", A.Data[Row * A.Cols + Col]);
        }
	printf("\n");
    }
}

matrix Transpose(matrix A)
{
    matrix Result = Matrix(A.Cols, A.Rows);
    for (int I = 0; I < A.Rows; ++I)
    {
	for (int J = 0; J < A.Cols; ++J) {
	    Result.Data[J * Result.Cols + I] = A.Data[I * A.Cols + J];
	}
    }
    return Result;
}

inline matrix
operator+(matrix A, matrix B)
{
    assert(A.Cols == B.Cols && A.Rows == B.Rows);
    matrix Result = Matrix(A.Rows, A.Cols);
    for(i32 Row = 0; Row < A.Rows; ++Row)
    {
        for(i32 Col = 0; Col < A.Cols; ++Col)
        {
            Result.Data[Row * Result.Cols + Col] =
            A.Data[Row * A.Cols + Col] +
            B.Data[Row * B.Cols + Col];
        }
    }

    return Result;
}


inline matrix
operator-(matrix A, matrix B)
{
    assert(A.Cols == B.Cols && A.Rows == B.Rows);
    matrix Result = Matrix(A.Rows, A.Cols);
    for(i32 Row = 0; Row < A.Rows; ++Row)
    {
        for(i32 Col = 0; Col < A.Cols; ++Col)
        {
            Result.Data[Row * Result.Cols + Col] =
            A.Data[Row * A.Cols + Col] -
            B.Data[Row * B.Cols + Col];
        }
    }

    return Result;
}

inline matrix
operator-(int a, matrix B)
{
    matrix Result = Matrix(B.Rows, B.Cols);
    for(i32 Row = 0; Row < B.Rows; ++Row)
    {
        for(i32 Col = 0; Col < B.Cols; ++Col)
        {
            Result.Data[Row * Result.Cols + Col] =
            1 -
            B.Data[Row * B.Cols + Col];
        }
    }

    return Result;
}

inline matrix
Hadamard(matrix A, matrix B)
{
    assert(A.Cols == B.Cols && A.Rows == B.Rows);
    matrix Result = Matrix(A.Rows, A.Cols);
    for(i32 Row = 0; Row < A.Rows; ++Row)
    {
        for(i32 Col = 0; Col < A.Cols; ++Col)
        {
            Result.Data[Row * Result.Cols + Col] =
            A.Data[Row * Result.Cols + Col] *
            B.Data[Row * Result.Cols + Col];
        }
    }

    return Result;
    
}

inline matrix
operator*(matrix A, matrix B)
{
    assert(A.Cols == B.Rows);
    matrix Result = Matrix(A.Rows, B.Cols);
    for(i32 Row = 0; Row < A.Rows; ++Row)
    {
        for(i32 Col = 0; Col < B.Cols; ++Col)
        {
            f32 sum = 0;
            for(i32 K=0; K < B.Rows; K++)
            {
                sum += A.Data[Row * A.Cols + K] * B.Data[K * B.Cols + Col];
            }
            Result.Data[Row * Result.Cols + Col] = sum;
        }
    }

    return Result;
}

inline matrix Scalar(matrix A, f32 Scalar)
{
    matrix Result = Matrix(A.Rows, A.Cols);
    for(i32 Row = 0; Row < A.Rows; ++Row)
    {
	for(i32 Col = 0; Col < A.Cols; ++Col)
	{
	    Result.Data[Row * Result.Cols + Col] = A.Data[Row * A.Cols + Col];
	}
    }

    for(i32 Row = 0; Row < A.Rows; ++Row)
    {
	for(i32 Col = 0; Col < A.Cols; ++Col)
	{
	    Result.Data[Row * Result.Cols + Col] *= Scalar;
	}
    }

    return Result;
}

inline matrix
operator*(f32 A, matrix B)
{
    return Scalar(B, A);
}

inline void Set(matrix* M, i32 Row, i32 Col, f32 Val)
{
    M->Data[Row * M->Cols + Col] = Val;
}
