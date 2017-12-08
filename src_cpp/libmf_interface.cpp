//
//  libmf_interface.cpp
//  
//
//  Created by Sam Royston on 9/28/16.
//  Copyright © 2016 Sam Royston. All rights reserved.
//

#include <iostream>
#include "mf.h"


bool is_functionality(std::string arg){
    return arg.compare("start-dump") == 0 || arg.compare("update") == 0
                                          || arg.compare("construct-matrix") == 0;
}

mf::mf_problem prob_from_sparse(const int nnx, float *X){
    size_t i;
    mf::mf_problem prob;
    prob.nnz = nnx;
    prob.m = 0;
    prob.n = 0;
    prob.R = nullptr;
    mf::mf_node *R = new mf::mf_node[prob.nnz];
    for(i=0; i < prob.nnz; i++){
        mf::mf_node N;
        
        N.u = (int)X[i * 3];
        N.v = (int)X[i * 3 + 1];
        N.r = X[i * 3 + 2];
        if(N.u+1 > prob.m)
            prob.m = N.u+1;
        if(N.v+1 > prob.n)
            prob.n = N.v+1;
        R[i] = N;
    }
    prob.R = R;
    return prob;
}

mf::mf_model assemble_model(mf::mf_int fun,
                            mf::mf_int m,
                            mf::mf_int n,
                            mf::mf_int k,
                            mf::mf_float b,
                            mf::mf_float *P,
                            mf::mf_float *Q)
{
    mf::mf_model model;
    model.k = k;
    model.m = m;
    model.n = n;
    model.b = b;
    model.fun = fun;
    model.P = P;
    model.Q = Q;
    return model;
}

#ifdef __cplusplus
extern "C" void train_interface(const int nnx, float *X, const char *path)
#else
void train_interface(const int nnx, float *X, const char *path)
#endif
{
    mf::mf_problem prob = prob_from_sparse(nnx, X);
    mf::mf_parameter param = mf::mf_get_default_param();
    const mf::mf_model *model = mf::mf_train(&prob, param);
    mf::mf_save_model(model, path);
    std::cout << "saved_model" << std::endl;
}

#ifdef __cplusplus
extern "C" mf::mf_model* fit_interface(const int nnx, float *X, mf::mf_parameter *param)
#else
mf::mf_model* fit_interface(const int nnx, float *X, mf::mf_parameter *param)
#endif
{
    mf::mf_problem prob = prob_from_sparse(nnx, X);
    mf::mf_model *out = mf::mf_train(&prob, *param);
    return out;
}

#ifdef __cplusplus
extern "C" mf::mf_model* train_valid_interface(const int nnx, const int nnxv,
                                      float *X, float *V, mf::mf_parameter *param)
#else
mf::mf_model* train_valid_interface(const int nnx, const int nnxv,
                           float *X, float *V, mf::mf_parameter *param)
#endif
{
    mf::mf_problem prob = prob_from_sparse(nnx, X);
    mf::mf_problem valid_prob = prob_from_sparse(nnxv, V);
    mf::mf_model *model = mf::mf_train_with_validation(&prob, &valid_prob,
                                                       *param);
    return model;
}

#ifdef __cplusplus
extern "C" mf::mf_double cross_valid_interface(const int nnx, float *X,
                                               mf::mf_parameter *param, int folds)
#else
mf::mf_double cross_valid_interface(const int nnx, float *X,
                                    mf::mf_parameter *param, int folds)
#endif
{
    const mf::mf_problem prob = prob_from_sparse(nnx, X);
    mf::mf_double score = mf::mf_cross_validation(&prob, folds, *param);
    return score;
}

#ifdef __cplusplus
extern "C" float* pred_model_interface(const int nnx,
                                    float *X,
                                    float *out,
                                    mf::mf_model *model)
#else
float* pred_model_interface(const int nnx,
                         float *X,
                         float *out,
                         mf::mf_model *model)
#endif
{
    for (int i = 0; i < nnx; i++){
        out[i] = mf::mf_predict(model, X[i], X[i + nnx]);
    }
    return out;
}


#ifdef __cplusplus
extern "C" float* get_P(float *out, mf::mf_model *model)
#else
float* get_P(float *out, mf::mf_model *model)
#endif
{
    for (int i = 0; i < model->m; i++){
        for(int j = 0; j < model->k; j++){
            int idx = i * model->k + j;
            out[idx] = model->P[idx];
        }
    }
    return out;
}

#ifdef __cplusplus
extern "C" float* get_Q(float *out, mf::mf_model *model)
#else
float* get_Q(float *out, mf::mf_model *model)
#endif
{
    for (int i = 0; i < model->n; i++){
        for(int j = 0; j < model->k; j++){
            int idx = i * model->k + j;
            out[idx] = model->Q[idx];
        }
    }
    return out;
}



    
