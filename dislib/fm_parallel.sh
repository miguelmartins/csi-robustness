#!/bin/bashhh
parallel -j 2 'CUDA_VISIBLE_DEVICES=$(({%} % 2)) uv run dislib/fm_pgd.py  --dataset {1} &> fm_{2}_{1}.txt' ::: shapes3d dsprites cars3d smallnorb::: v3conv v3vit v2vit

parallel -j 4 'CUDA_VISIBLE_DEVICES=$(({%} % 2)) uv run dislib/pgd.py --rep {1} --aug {2} --target {4} --target_idx {5} --dataset {3}  &> out_{1}_{2}_{3}_{4}_{5}.txt' ::: 0 1 2 3 ::: none crop sup sup2 simclr2 ::: shapes3d dsprites cars3d smallnorb ::: continuous manifold other categorical ::: 0 1 2 3  


