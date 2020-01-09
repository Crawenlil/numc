#include "numc.h"


template<typename T>
class Linear{
    private:
        Matrix<T> W;
        Matrix<T> b;
        void backprop();
    public:
        // constructors
        Linear();
        Linear(const Matrix<T> &_W, const Matrix<T> &_b);
        Linear(size_t input_dim, size_t output_dim);
        // destructor
        ~Linear();
        // copy constructor
        Linear<T>& Linear(const Linear<T> other);
        // assignment 
        Linear<T>& operator=()(const Linear<T> other);
        Matrix<T>& operator()(const Matrix<T> input) const;
};


template<typename T>
Linear<T>::Linear() {}

template<typename T>
Linear<T>::Linear(const Matrix<T> &_W, const Matrix<T> &_b) {
    W = _W;
    b = _b;
}

template<typename T>
Linear<T>::Linear(size_t input_dim, size_t output_dim) {
    W = Matrix(input_dim, output_dim);
    b = Matrix(1, output_dim);
    W.initNormal();
    b.initNormal();
}

template<typename T>
Linear<T>::~Linear(){}

template<typename T>
Linear<T>& Linear<T>::Linear(const Linear<T> other) {
    W = other.W;
    b = other.b;
}

template<typename T>
Linear<T>& Linear<T>::operator=()(const Linear<T> other) {
    W = other.W;
    b = other.b;
}

template<typename T>
Matrix<T>& Linear<T>::operator()(const Matrix<T> input) const {
    Matrix<T> nb = b.repeat(input.getRows());
    return input.mm(W) + nb;
}
