for dataset in adni_multi_atlas #abide_full_multi_atlas matai_multi_atlas  ppmi_multi_atlas
  do
    for i in 0.01 0.1 1.0 10.0
    do
  for d in 1e-3 1e-4 1e-5 #0.01 0.1 1.0 10.0
  do
    for lm in 1.0 10.0 100.0
    do
        model="configs/"${dataset}"/TUs_graph_classification_AIDFusion_"${dataset}"_100k.json"
        echo ${model}
        CUDA_LAUNCH_BLOCKING=0 python3 main.py --gpu_id 0 --L 1 --node_feat_transform pearson --max_time 60 --config $model --edge_ratio 0.2 --lambda1 $i --lambda2 $lm --lambda3 $d --lambda4 1.0 --dropout 0.5
    done
    done
  done
done
