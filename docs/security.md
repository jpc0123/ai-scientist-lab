# 安全边界

Scientist Lab 默认偏保守，产品安全不等于「按钮藏起来」。

## 硬约束

| 项 | 默认 |
|----|------|
| LLM | Mock，不连公网模型 |
| 网络 | 默认关闭外连科研调用 |
| Shell | 无任意 Shell |
| 主工作区 | UI 不可直接写主树 |
| 补丁 | 仅沙箱；合入需人工审批 |
| Git | 无自动 push；回滚仅 revert |
| Claim | Claim Gate；不自动升格科学主张 |
| 重启恢复 | 不自动重跑昂贵实验 |

## 查看当前态势（v2.0.9）

Web：「系统」→「安全边界」  
API：`GET /api/v1/system/security`

## 人工审批

Plan Candidate、Iteration、补丁、Merge 等关键步骤需人工确认。前端确认框不能替代后端校验。

## 错误体验（v2.0.9）

API 统一返回：

```json
{
  "error": {
    "type": "protocol_mismatch",
    "code": "protocol_mismatch",
    "message": "...",
    "retryable": false,
    "details": {},
    "suggested_action": "..."
  }
}
```

前端展示：错误是什么 / 为什么 / 你可以做什么 / 是否可重试。  
**不会**把 Python traceback 直接展示给用户（仅服务日志）。

## 演示项目

`demo-create` 只播种结构，不代表已验证的科学结论。RGB-T Debug 明确标注「无自动科学宣称」。

## 多用户 / 云

v2.0 **不做**多租户 SaaS、团队权限、公有云托管。本机单用户使用。
