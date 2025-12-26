# MindStudio Modeling

## 简介

**MindStudio Modeling** 是 MindStudio 建模寻优工具，评估模型及服务化场景下的理论性能，并在此基础上寻找性能较优的部署策略等参数。

## 目录结构
关键目录如下，详细目录介绍参见[项目目录](./docs/zh/dir_structure.md)。
```
├── README.md                        # 项目说明文档
├── LICENSE.md                       # 项目许可证文档
├── docs/                            # 文档目录
│   ├── images/                      # 图片资源
│   └── zh/                          # 中文文档
└── src/                             # 源代码目录
    └── optimize/                    # 自动寻优相关模块
```

## 环境部署

### 环境和依赖

- 硬件环境请参见《[昇腾产品形态说明](https://www.hiascend.com/document/detail/zh/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html)》。

- 软件环境请参见《[CANN 软件安装指南](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/softwareinst/instg/instg_quick.html?Mode=PmIns&InstallType=local&OS=openEuler&Software=cannToolKit)》安装昇腾设备开发或运行环境，即Toolkit软件包。

以上环境依赖请根据实际环境选择适配的版本。

### 工具安装
安装MindStudio Modeling工具，详情请参见[安装指南](./docs/zh/serviceparam_optimizer.md/#使用前准备)。

## 功能介绍
- [服务化自动寻优工具](./docs/zh/serviceparam_optimizer.md)

    支持对 `MindIE` 和 `VLLM` 进行自动寻优，获取符合时延要求的最佳吞吐参数组合。

## 免责声明

- 本工具仅供调试和开发之用，使用者需自行承担使用风险，并理解以下内容：

  - [X] 数据处理及删除：用户在使用本工具过程中产生的数据属于用户责任范畴。建议用户在使用完毕后及时删除相关数据，以防泄露或不必要的信息泄露。
  - [X] 数据保密与传播：使用者了解并同意不得将通过本工具产生的数据随意外泄或传播。对于由此产生的信息泄露、数据泄露或其他不良后果，本工具及其开发者概不负责。
  - [X] 用户输入安全性：用户需自行保证输入的命令行的安全性，并承担因输入不当而导致的任何安全风险或损失。对于由于输入命令行不当所导致的问题，本工具及其开发者概不负责。
- 免责声明范围：本免责声明适用于所有使用本工具的个人或实体。使用本工具即表示您同意并接受本声明的内容，并愿意承担因使用该功能而产生的风险和责任，如有异议请停止使用本工具。
- 在使用本工具之前，请**谨慎阅读并理解以上免责声明的内容**。对于使用本工具所产生的任何问题或疑问，请及时联系开发者。

## License

MindStudio Modeling产品的木兰宽松许可证，具体请参见[LICENSE](LICENSE)文件。

msProf工具docs目录下的文档适用CC-BY 4.0许可证，具体请参见[LICENSE](docs/LICENSE)。

## 安全声明

有关MindStudio Modeling产品的安全加固信息、公网地址信息及通信矩阵等内容，请参见[MindStudio Modeling安全声明](./docs/zh/security_statement.md)。

## 建议与交流

欢迎大家为社区做贡献。如果有任何疑问或建议，请提交[Issues]((https://gitcode.com/Ascend/msmodeling/issues)，我们会尽快回复。感谢您的支持。

## 致谢

MindStudio Modeling由华为公司的下列部门联合贡献：

- 昇腾计算MindStudio开发部
- 2012软件工程实验室

感谢来自社区的每一个PR，欢迎贡献MindStudio Modeling！
