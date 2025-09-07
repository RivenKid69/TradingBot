cdef int compute_n_features(list layout)
cdef void build_observation_vector_c(const EnvState* state, MarketSimulatorWrapper* market, float[::1] ext_norm_cols, float[::1] out_vec) nogil
