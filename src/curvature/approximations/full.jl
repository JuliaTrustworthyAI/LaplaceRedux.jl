"""
    full(curvature::CurvatureInterface, d::Tuple; batched::Bool=false)

Compute the full approximation, for either a single input-output datapoint or a batch of such. 
"""
function full(curvature::CurvatureInterface, d::Tuple; batched::Bool=false)
    if batched
        full_batched(curvature, d)
    else
        full_unbatched(curvature, d)
    end
end
