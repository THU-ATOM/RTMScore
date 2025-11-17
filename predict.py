#!/usr/bin/env python3
"""
RTMScore 虚拟筛选预测脚本

此脚本在 Docker 容器中运行，用于对蛋白质-配体对进行打分
"""

import argparse
import json
import sys
import os
import gzip
from pathlib import Path
import torch as th
import numpy as np
from rdkit import Chem, RDLogger
from torch.utils.data import DataLoader

# 禁用 RDKit 警告
RDLogger.DisableLog('rdApp.*')

# 禁用 Python 输出缓冲
sys.stdout = sys.stdout
sys.stderr = sys.stderr

# 添加 RTMScore 到 Python 路径
sys.path.append("/app")
sys.path.append("/app/RTMScore")

from RTMScore.data.data import VSDataset
from RTMScore.model.utils import collate, run_an_eval_epoch
from RTMScore.model.model2 import RTMScore as RTMScoreModel, DGLGraphTransformer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='RTMScore 虚拟筛选预测脚本')
    parser.add_argument('-i', '--input', type=str, required=True, help='输入 JSON 文件路径')
    parser.add_argument('-o', '--output', type=str, required=True, help='输出目录路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点目录')
    parser.add_argument('--batch-size', type=int, default=128, help='批次大小')
    parser.add_argument('--cutoff', type=float, default=10.0, help='定义口袋的距离截断值')
    parser.add_argument('--dist-threshold', type=float, default=5.0, help='距离阈值')
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载器的工作进程数')
    
    return parser.parse_args()


def load_input_data(input_path):
    """加载输入数据"""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def read_sdf_molecules(sdf_path):
    """读取 SDF 文件中的分子"""
    molecules = []
    
    # 处理 .sdfgz 格式
    if sdf_path.endswith('.sdfgz') or sdf_path.endswith('.sdf.gz'):
        with gzip.open(sdf_path, 'rb') as f:
            suppl = Chem.ForwardSDMolSupplier(f)
            for mol in suppl:
                if mol is not None:
                    molecules.append(mol)
    else:
        # 处理普通 .sdf 格式
        suppl = Chem.SDMolSupplier(sdf_path)
        for mol in suppl:
            if mol is not None:
                molecules.append(mol)
    
    return molecules


def prepare_ligand_file(actives_path, decoys_path, output_path):
    """
    合并 actives 和 decoys 为一个 SDF 文件，并返回标签列表
    
    Returns:
        labels: 标签列表 (1 for active, 0 for decoy)
    """
    labels = []
    all_mols = []
    
    # 读取 actives
    if actives_path and os.path.exists(actives_path):
        print(f"  读取 actives: {actives_path}", flush=True)
        actives = read_sdf_molecules(actives_path)
        print(f"    找到 {len(actives)} 个 active 分子", flush=True)
        all_mols.extend(actives)
        labels.extend([1] * len(actives))
    
    # 读取 decoys
    if decoys_path and os.path.exists(decoys_path):
        print(f"  读取 decoys: {decoys_path}", flush=True)
        decoys = read_sdf_molecules(decoys_path)
        print(f"    找到 {len(decoys)} 个 decoy 分子", flush=True)
        all_mols.extend(decoys)
        labels.extend([0] * len(decoys))
    
    # 写入合并后的 SDF 文件
    if all_mols:
        writer = Chem.SDWriter(output_path)
        for mol in all_mols:
            writer.write(mol)
        writer.close()
        print(f"  合并后的配体文件: {output_path} (共 {len(all_mols)} 个分子)", flush=True)
    else:
        print(f"  警告: 未找到任何配体分子", flush=True)
    
    return labels


def scoring(prot, lig, modpath, cutoff=10.0, dist_threshold=5.0, 
            gen_pocket=False, reflig=None, batch_size=128, num_workers=4):
    """
    RTMScore 打分函数
    
    Args:
        prot: 蛋白质文件路径 (.pdb)
        lig: 配体文件路径 (.sdf)
        modpath: 预训练模型路径
        cutoff: 定义口袋的距离截断值
        dist_threshold: 距离阈值
        gen_pocket: 是否从蛋白质文件生成口袋
        reflig: 参考配体用于定义口袋
        batch_size: 批次大小
        num_workers: 数据加载器的工作进程数
    
    Returns:
        mol_ids: 分子 ID 列表
        scores: 预测分数列表
    """
    device = 'cuda' if th.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}", flush=True)
    
    # 创建数据集
    print(f"创建数据集...", flush=True)
    print(f"  蛋白质: {prot}", flush=True)
    print(f"  配体: {lig}", flush=True)
    print(f"  cutoff: {cutoff}", flush=True)
    print(f"  gen_pocket: {gen_pocket}", flush=True)
    if reflig:
        print(f"  reflig: {reflig}", flush=True)
    
    data = VSDataset(
        ligs=lig,
        prot=prot,
        cutoff=cutoff,
        gen_pocket=gen_pocket,
        reflig=reflig,
        explicit_H=False,
        use_chirality=True,
        parallel=False
    )
    
    test_loader = DataLoader(
        dataset=data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate
    )
    
    print(f"数据集创建完成，共 {len(data)} 个样本", flush=True)
    
    # 定义模型参数
    model_params = {
        "num_node_featsl": 41,
        "num_node_featsp": 41,
        "num_edge_featsl": 10,
        "num_edge_featsp": 5,
        "hidden_dim0": 128,
        "hidden_dim": 128,
        "n_gaussians": 10,
        "dropout_rate": 0.10,
    }
    
    # 创建配体模型
    ligmodel = DGLGraphTransformer(
        in_channels=model_params["num_node_featsl"],
        edge_features=model_params["num_edge_featsl"],
        num_hidden_channels=model_params["hidden_dim0"],
        activ_fn=th.nn.SiLU(),
        transformer_residual=True,
        num_attention_heads=4,
        norm_to_apply='batch',
        dropout_rate=0.15,
        num_layers=6
    )
    
    # 创建蛋白质模型
    protmodel = DGLGraphTransformer(
        in_channels=model_params["num_node_featsp"],
        edge_features=model_params["num_edge_featsp"],
        num_hidden_channels=model_params["hidden_dim0"],
        activ_fn=th.nn.SiLU(),
        transformer_residual=True,
        num_attention_heads=4,
        norm_to_apply='batch',
        dropout_rate=0.15,
        num_layers=6
    )
    
    # 创建 RTMScore 模型
    model = RTMScoreModel(
        ligmodel, protmodel,
        in_channels=model_params["hidden_dim0"],
        hidden_dim=model_params["hidden_dim"],
        n_gaussians=model_params["n_gaussians"],
        dropout_rate=model_params["dropout_rate"],
        dist_threhold=dist_threshold
    ).to(device)
    
    # 加载模型权重
    print(f"加载模型权重: {modpath}", flush=True)
    checkpoint = th.load(modpath, map_location=th.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 运行推理
    print(f"开始推理...", flush=True)
    preds = run_an_eval_epoch(
        model, test_loader,
        pred=True,
        dist_threhold=dist_threshold,
        device=device
    )
    
    print(f"推理完成，获得 {len(preds)} 个预测分数", flush=True)
    
    return data.ids, preds


def main():
    args = parse_args()
    
    print("=" * 80, flush=True)
    print("RTMScore 虚拟筛选预测", flush=True)
    print("=" * 80, flush=True)
    print(f"输入文件: {args.input}", flush=True)
    print(f"输出目录: {args.output}", flush=True)
    print(f"模型目录: {args.checkpoint}", flush=True)
    print(f"批次大小: {args.batch_size}", flush=True)
    print(f"距离截断: {args.cutoff}", flush=True)
    print(f"距离阈值: {args.dist_threshold}", flush=True)
    print("=" * 80, flush=True)
    
    # 加载输入数据
    input_data = load_input_data(args.input)
    print(f"\n加载了 {len(input_data)} 个目标", flush=True)
    
    # 确保输出目录存在
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找模型权重文件
    checkpoint_dir = Path(args.checkpoint)
    model_path = checkpoint_dir / "rtmscore_model1.pth"
    if not model_path.exists():
        print(f"错误: 模型文件不存在: {model_path}", flush=True)
        sys.exit(1)
    
    # 对每个目标进行预测
    for i, target_data in enumerate(input_data):
        target_name = target_data["name"]
        print(f"\n{'=' * 80}", flush=True)
        print(f"处理目标 [{i+1}/{len(input_data)}]: {target_name}", flush=True)
        print(f"{'=' * 80}", flush=True)
        
        try:
            # 确定使用 pocket 还是 receptor
            use_pocket = target_data.get("use_pocket", False)
            
            if use_pocket:
                prot_path = target_data.get("pocket_path")
                gen_pocket = False
                reflig = None
                print(f"使用 pocket 文件: {prot_path}", flush=True)
            else:
                prot_path = target_data.get("receptor_path")
                gen_pocket = True
                reflig = target_data.get("ligand_path")
                print(f"使用 receptor 文件: {prot_path}", flush=True)
                print(f"参考配体: {reflig}", flush=True)
            
            if not prot_path or not os.path.exists(prot_path):
                print(f"错误: 蛋白质文件不存在: {prot_path}", flush=True)
                continue
            
            # 准备配体文件
            actives_path = target_data.get("actives_path")
            decoys_path = target_data.get("decoys_path")
            
            # 合并 actives 和 decoys
            combined_lig_path = output_dir / f"{target_name}_ligands.sdf"
            labels = prepare_ligand_file(actives_path, decoys_path, str(combined_lig_path))
            
            if not os.path.exists(combined_lig_path):
                print(f"错误: 无法创建配体文件", flush=True)
                continue
            
            # 运行打分
            mol_ids, scores = scoring(
                prot=prot_path,
                lig=str(combined_lig_path),
                modpath=str(model_path),
                cutoff=args.cutoff,
                dist_threshold=args.dist_threshold,
                gen_pocket=gen_pocket,
                reflig=reflig,
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )
            
            # 保存结果
            result = {
                "target": target_name,
                "mol_ids": mol_ids,
                "scores": scores.tolist() if isinstance(scores, np.ndarray) else scores,
                "labels": labels
            }
            
            output_file = output_dir / f"{target_name}.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            print(f"\n✓ 结果已保存: {output_file}", flush=True)
            print(f"  分子数量: {len(scores)}", flush=True)
            print(f"  分数范围: [{min(scores):.4f}, {max(scores):.4f}]", flush=True)
            
        except Exception as e:
            print(f"\n✗ 处理 {target_name} 时出错: {str(e)}", flush=True)
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'=' * 80}", flush=True)
    print("所有目标处理完成", flush=True)
    print(f"{'=' * 80}", flush=True)


if __name__ == "__main__":
    main()
