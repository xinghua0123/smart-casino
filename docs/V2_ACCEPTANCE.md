# 2.0：验收与运行指南

状态：本地实现与验证已完成；用户于 2026-09-06 授权提交、创建 `2.0` 标签并推送仓库。`1.0` 固定指向 `6d10adf0187f6f5569be57aff559fae09e13bb01`。

## 打开与启动

看板：<http://localhost:8501>。运营 API 健康检查：<http://localhost:8090/health>。

本机开发环境使用已有的 Python 依赖镜像，并通过只读目录挂载运行当前工作区代码：

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --no-build
```

首次在没有这些本地镜像的机器部署，使用标准构建：

```bash
docker compose up --build -d
```

此次本机验收采用前一种方式，外部 Python 基础镜像元数据请求曾长时间无响应；标准联网构建没有在此次环境中完成验证。开发覆盖文件依赖本机已有 `smart-casino-floor-dashboard` 和 `smart-casino-floor-data-producer` 镜像，不能单独用于全新机器。

模拟时钟默认 **10 倍速**：1 个模拟分钟约 6 秒。一键批准直接在模拟器应用（界面通常几秒后收到流回执）；5 分钟效果观察约 30 秒，15 分钟约 90 秒。页面自动刷新独立于聊天。

## 建议验收顺序

### 新手引导与英文界面

- 首次访问自动显示英文引导；侧栏 **Start guided tour** 可随时重新开始。
- 使用 **Next / Back / Skip tour**，或按 Escape 关闭。完成／跳过后，同一浏览器不会重复自动弹出。
- 当前简化引导共 6 步，覆盖欢迎、地图、客流场景、审批派发、效果观察与证据；高级规划默认折叠，不再纳入引导。
- 实时刷新和正常表单操作不会重置引导进度；引导本身不会触发场景、批准或派发任务。
- “Give the floor a goal”的默认示例、追问示例与错误提示均为英文。

### A. 自动建议与一键批准

1. 点 **Reset floor scenario**，等 LIVE 后点 **Dining group arrives**。
2. 观察 **Guests waiting** 与 **Estimated wait** 上升，顶部提示 Action center 有建议。
3. 进入 **Action center**，应看到开桌增员、热门桌提高最低限额等可行建议。
4. 点开桌建议的 **Approve**。不需要选择负责人，也不需要 Execute；几秒后显示 Applied，地图桌台开放、排队人数减少。
5. 检查卡片记录的现场生效前／后排队人数与预计等待时间。人员自动分配，资源仍须真实可用。
6. 点调价建议的 **Approve**，确认桌台实际最低限额改变。提高限额不一定减少排队，界面不得承诺必然改善。
7. +5／+15 模拟分钟后的观察会自动保存，不需要继续点击。

顶部 Estimated wait 根据队列和开放容量估算；实际已等待时长保留在 Details & history。高级目标、约束和预演仍可在 Advanced planning 中展开。

### B. 资源变化与数据检查

1. 有开桌建议但尚未批准时，点 **Reassign relief dealers**。
2. 不可行的旧建议应失效，不能虚构可用人员；调价建议仍按各自条件判断。
3. 批准与模拟器消费之间若发生资源丢失，应失败而不是误报成功。
4. 重复点击批准，或服务重启后重试，同一命令最多应用一次。

### C. 数据中断与恢复

1. 点 **Interrupt telemetry**，等待约 15 秒进入 STALE。
2. 保留最后已知值，并阻止生成新计划、批准和派发；不把断流解释为零客流。
3. 点 **Resume telemetry**，回到 LIVE。
4. 观察时间段如存在中断，结果应标注数据缺口；历史命令不得重复执行。

### D. 证据、误差与已有功能

- **Evidence & learning** 展示流快照序号、业务信号、员工资源、物理事件和计划记录。
- 保存方案后等至少 15 个模拟分钟，查看预测队列与实际队列误差。
- 基线期间如执行了动作或有数据缺口，应排除该比较；这里只做预测回看，不声称因果收益。
- 点击 **Player analytics & chat**，确认 VIP 雷达、Theo 图表、历史推荐和原有聊天入口仍可用。
- 历史玩家页跨场景保留数据，人数不能当作当前在座人数；当前值以运营地图为准。

## AI 模式

- 未配置密钥：明确显示 **Template parser**。支持指定的中英文模板和手动约束，不能将任意文本都视作理解成功。
- 配置 LLM：侧栏 **AI connection** 可填 OpenAI／Claude／OpenRouter 的密钥、模型和可选网关。也可通过服务环境变量 `OPS_LLM_API_KEY`、`OPS_LLM_PROVIDER`、`OPS_LLM_MODEL`、`OPS_LLM_BASE_URL` 配置。
- Claude 应选择该账户实际可用的 Claude 模型；OpenRouter 使用其模型标识。
- 密钥不会写入任务、计划或事件账本。任务执行不由 LLM 直接写 SQL，数值与资源校验由受控业务模块完成。
- 本次测试覆盖模拟 LLM 返回、无效字段、隐式放宽约束及请求失败回退。没有配置真实供应商密钥，因此未验证外部供应商的实际联网调用；验收时可输入自己的配置进行该项检查。

## 实现与数据

- `casino/`：物理状态、座位／队列、人员、命令准备和生效。
- `data_producer/floor_simulator.py`：持久化模拟器、事件生成、命令消费、Kafka 发布。
- `risingwave_sql/06_operations.sql`：运营源、最新全场快照、桌台与区域状态。
- `operations_service/`：RisingWave 快照消费、SQLite WAL 事务账本、方案模拟、目标解析与 API。
- `dashboard/app.py`：运营工作区；`dashboard/analytics.py` 保留玩家分析与聊天。
- `operations-state`、`simulator-state`、`risingwave-state` 三个卷保存各自状态。Reset 只切换模拟场景，保留历史任务、方案与玩家事件。
- 初始化脚本记录迁移版本；重复启动不会重建或清空行动历史。首次升级只重建原有 floor 派生视图。

## 验证命令

当前简化流程新增 6 项单元检查（总计 31 项），真实流验证见 `tests/live_quick_approval.py` 和 `docs/QUICK_APPROVAL_RESULTS.json`。

此次已通过 25 项单元测试、15 项真实链路验收检查，以及运营页／玩家分析页回归。浏览器已验证地图点选、未来时间切换、目标追问和页面导航。

英文界面与引导更新：已验证英文默认目标、`Exclude B08` 追问、英文错误提示；浏览器走通 10 步引导、Back、Finish、Skip、Escape 和重新开始，并在 508px 窄屏验证侧栏展开／收起及调整尺寸后按钮仍可操作。

```bash
python3 -B -m unittest discover -s tests -v
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec -T dashboard python < tests/dashboard_smoke.py
python3 -B tests/live_acceptance.py
```

`live_acceptance.py` 操作本地模拟场景，并重启模拟器／运营服务验证恢复；不删除历史账本。最近结果保存在 `docs/LIVE_TEST_RESULTS.json`。

## 首版边界

- 每个方案比较一个桌台动作；这是受约束的候选搜索，不是全场多动作最优解的证明。
- 到场率、等待模型、餐饮转化和成本为可解释的演示假设，尚未用真实赌场数据校准。
- 首版通过记录结果和预测误差支持后续学习，不自动训练或发布经营策略。
- 操作人员是演示身份，尚无企业身份认证；当前 API 只映射到本机地址。
- 真实 WDTS／CMS 接入、生产可用性和真实收益验证属于后续集成。
