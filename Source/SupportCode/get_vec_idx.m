function vec_idx = get_vec_idx( value, vector )
%GET_VEC_IDX Get the index of a value in a vector
%   In: value, vector, out: index

[ val, vec_idx ] = min( abs( value - vector ) );

end

