
## 中国老年人抑郁倾向的复合风险、条件因果效应与分层治理 —— 基于 CHARLS 数据的因果机器学习研究

> Compound Risks, Conditional Causal Effects, and Stratified Governance of Depressive Symptoms among Older Adults in China: A Causal Machine Learning Study Using CHARLS Data

本仓库是<中国老年人抑郁倾向的复合风险、条件因果效应与分层治理 —— 基于 CHARLS 数据的因果机器学习研究>的完整数据处理、建模与复现代码。研究基于 CHARLS 2011—2020 五期面板，在"社会资本-贫困健康-健康行为"三大理论整合框架下，借助可解释机器学习识别核心暴露因素，进而通过面板固定效应、双重机器学习与因果随机森林对老年抑郁倾向的成因、条件因果效应及其异质性进行系统估计，并补充复合风险与协同关联分析，最终落到分层治理的政策含义。

---

## 1. 研究问题

围绕5个彼此关联的具体问题展开。

第一，中国老年人抑郁倾向呈现出怎样的时空分布特征？

第二在较高维度的候选变量中，哪些因素是预测老年抑郁的关键变量？

第三，社会参与、多病共存和睡眠异常这三类核心因素，是否对老年抑郁具有更严格意义上的效应？如果存在，这种效应究竟有多大，且是否在不同性别、城乡和区域群体之间存在系统差异？

第四，社会参与不足、多病共存和睡眠异常三类风险因素之间是否存在累积叠加、协同关联或条件分化？

第五，基于上述效应识别、复合风险分析和异质性分析结果，应当如何形成更有针对性的老年抑郁缓解路径？

## 2. 数据来源

中国健康与养老追踪调查（China Health and Retirement Longitudinal Study, CHARLS）2011、2013、2015、2018、2020 共五期非平衡面板。原始数据由北京大学国家发展研究院提供，本仓库不分发数据，运行前需要从 [CHARLS 官网](https://charls.pku.edu.cn/) 申请下载。

经清洗后的两套分析样本：

| 样本 | 用途 | 观测数 | 个体数 |
|---|---|---:|---:|
| FE 主样本（≥2 波次） | 面板固定效应基准回归与交互效应分析 | 36,738 | 10,801 |
| 因果样本（lagged 配对） | DML 平均因果效应、CRF 异质性、亚组协同 | 24,616 | 10,354 |

结果变量为 CES-D10 抑郁自评得分（连续，0-30），核心处理变量为社会参与指数、慢性病数量（多病共存）与睡眠时长（睡眠异常）。

## 3. 研究设计

本研究采用"理论预期-数据探索-理论校验-因果验证"的双重驱动路径：理论预期来自社会资本理论、贫困-健康理论、健康行为理论的整合框架；数据探索基于 LASSO、随机森林、XGBoost 三种学习器与 SHAP 可解释性分析的高维筛选；理论校验对 XML 筛选结果进行学理验证和过滤；因果验证则由 FE/DML/CRF 三阶段构成，并补充复合风险与协同关联分析以回应风险维度交错叠加的现实复杂性。

完整研究技术路线图：

![研究技术路线图](11_figures_for_paper/technical_roadmap.png)



## 4. 方法与模型

主分析采用三种互补的因果识别方法：

**面板固定效应（FE）** 用于在控制个体不可观测异质性后估计三个处理变量与抑郁得分之间的稳健关联。标准误按个体聚类，控制变量为人口学、家庭与经济基本特征。FE 估计结果在本研究中表述为"控制个体固定效应后的关联"，不作因果声明。

**双重机器学习（DML, Chernozhukov et al. 2018）** 在条件独立假设下估计平均因果效应。第一阶段 nuisance function 采用 XGBoost（n_estimators=500, max_depth=5），5 折交叉拟合重复 3 次取中位数。连续处理变量用部分线性回归（PLR），二元处理变量用交互回归模型（IRM）。处理变量取 t-1 期值，结果变量取 t 期值，以构造时序合理的 lagged 设计。

**因果随机森林（CRF, Wager & Athey 2018; Athey, Tibshirani & Wager 2019）** 用于估计个体异质性因果效应。报告 ATE、ITE 分布、GATE 分组结果、BLP 检验、变量重要度与高受益人群画像，并通过三种异质性证据的一致性比较进行三角验证。

补充分析包括：复合风险变量构建与累积风险阶梯描述、FE 复合风险交互三模型（M1/M2/M3）与 Wald 联合显著性检验、DML 三组亚组协同估计与 nuisance fit 诊断。详见 `scripts/s6_*.py` 系列脚本。

## 5. 仓库结构

├── data/                          # 数据目录（不入库，需自行准备）
│   └── preprocess_outputs/
│       └── data1_modeling_ready.xlsx
│
├── scripts/                       # 主分析脚本，按阶段命名
│   ├── s1_.py                    # 数据预处理与变量构建
│   ├── s2_.py                    # 描述统计与时空分布
│   ├── s3_.py                    # 面板固定效应（FE）
│   ├── s4_.py                    # 可解释机器学习筛选（LASSO/RF/XGB + SHAP）
│   ├── s5_.py                    # 双重机器学习（DML）
│   ├── s6_01_crf_all.py          # 因果随机森林CRF
│   ├── s6_02_cumulative_risk_analysis.py     # 复合风险分析 A
│   ├── s6_03_fe_interaction_models.py        # 复合风险分析 B
│   ├── s6_04_dml_subgroup_synergy.py         # 复合风险分析 C
│   ├── s7_.py                    # 内生性讨论
│   ├── s8_.py                    # 稳健性检验
│   ├── s9_.py                    # 因果随机森林（CRF）与异质性
│   └── composite_risk.py      # 复合风险变量统一构建工具
│

└── README.md

## 6. 复现指南

**环境要求**：Python 3.10 或更高，建议使用虚拟环境。主要依赖：`pandas`、`numpy`、`scipy`、`scikit-learn`、`xgboost`、`linearmodels`、`doubleml`、`econml`、`shap`、`matplotlib`、`openpyxl`。

**安装**：

```bash
git clone https://github.com/ailiwood/Determinants-Heterogeneous-Causal-Effects-and-Mitigation-Pathways-of-Late-Life-Depression.git
cd Determinants-Heterogeneous-Causal-Effects-and-Mitigation-Pathways-of-Late-Life-Depression

```
**运行顺序**：先按 `s1` → `s2` → `s3` → `s4` 完成数据准备、描述统计、FE 基准与变量筛选；再按 `s5`（DML）→ `s7`（内生性）→ `s8`（稳健性）→ `s9`（CRF）完成主分析；最后运行 `s6_02`、`s6_03`、`s6_04` 完成复合风险补充分析。所有脚本均可独立运行，但跨阶段依赖中间产出文件。


## 7.  许可

代码部分采用 MIT 许可证。论文文本与图表的著作权归作者所有。

## 8. 联系

如对方法实现或结果复现有疑问，欢迎通过 GitHub Issues 提交问题。
