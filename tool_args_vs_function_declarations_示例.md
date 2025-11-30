# tool_args 与 function_declarations 的区别详解

## 一、基本概念

### 1. `tool_args` - 用户提取的参数值
- **来源**：从 LLM 的 `function_call` 中提取，基于用户描述和工具指令
- **内容**：只包含 LLM **认为用户想要使用的参数及其值**
- **特点**：
  - 只包含用户明确提到或 LLM 推断出的参数
  - 不包含有默认值的参数（除非用户明确指定）
  - 不包含可选参数（除非用户明确指定）

### 2. `function_declarations` - 工具的完整 Schema 定义
- **来源**：工具的完整 JSON Schema 定义（从 MCP Tool 获取）
- **内容**：包含工具的**所有参数**的完整定义
- **特点**：
  - 包含所有参数的名称、类型、描述、默认值、是否必需等信息
  - 是工具的"说明书"，定义了工具的所有可能参数

## 二、实际例子（基于日志）

### 示例：`optimize_structure` 工具

#### 阶段1：LLM 提取的初始 `tool_args`（从 function_call）

```python
# 从日志第161行可以看到，LLM 只提取了用户明确提到的参数：
tool_call_info = {
    'tool_name': 'optimize_structure',
    'tool_args': {
        'input_structure': 'https://bohrium.oss-cn-zhangjiakou.aliyuncs.com/.../Cu_bulk.cif',
        'model_path': 'https://bohrium.oss-cn-zhangjiakou.aliyuncs.com/.../dpa-2.4-7M.pt',
        'relax_cell': False
    },
    'missing_tool_args': []
}
```

**说明**：
- LLM 只提取了用户描述中明确提到的参数：
  - `input_structure`：用户提供了结构文件 URL
  - `model_path`：LLM 推断出需要模型路径
  - `relax_cell`：LLM 推断出需要设置（设为 False）
- **没有包含**其他有默认值的参数（如 `head`, `force_tolerance`, `max_iterations`）

#### 阶段2：`function_declarations` 的完整定义

```python
# function_declarations 包含工具的完整 Schema：
function_declarations = [
    {
        'name': 'optimize_structure',
        'description': 'Perform geometry optimization...',
        'parameters': {
            'type': 'object',
            'required': ['input_structure', 'model_path'],
            'properties': {
                'input_structure': {
                    'type': 'string',
                    'description': 'Input structure file URL or path'
                },
                'model_path': {
                    'type': 'string',
                    'description': 'Path to the DPA model file'
                },
                'head': {
                    'type': 'string',
                    'description': 'Model head type',
                    'default': 'Omat24'  # 有默认值
                },
                'force_tolerance': {
                    'type': 'number',
                    'description': 'Force convergence tolerance',
                    'default': 0.01  # 有默认值
                },
                'max_iterations': {
                    'type': 'integer',
                    'description': 'Maximum optimization iterations',
                    'default': 100  # 有默认值
                },
                'relax_cell': {
                    'type': 'boolean',
                    'description': 'Whether to relax cell parameters',
                    'default': False  # 有默认值
                },
                'executor': {
                    'type': 'object',
                    'description': 'Executor configuration'
                },
                'storage': {
                    'type': 'object',
                    'description': 'Storage configuration'
                }
            }
        }
    }
]
```

**说明**：
- 包含了工具的**所有参数**定义
- 每个参数都有完整的类型、描述、默认值等信息
- 明确标注了哪些是必需参数（`required`）

#### 阶段3：通过 `update_tool_call_info_with_function_declarations` 补充

```python
# 函数会遍历 function_declarations 中的所有参数，补充缺失的参数：

# 处理逻辑：
for param_name, param_schema in properties.items():
    # 跳过系统参数
    if param_name in ['executor', 'storage']:
        continue

    # 如果参数已经在 tool_args 中，跳过（保留用户/LLM 的值）
    if param_name in tool_args:
        continue

    # 如果参数有默认值，添加到 tool_args
    if 'default' in param_schema:
        tool_args[param_name] = param_schema['default']

    # 如果是必需参数但没有值，添加到 missing_tool_args
    elif param_name in required_params:
        missing_tool_args.append(param_name)

    # 可选参数且没有默认值，也添加到 missing_tool_args
    else:
        missing_tool_args.append(param_name)
```

**结果**（从日志第172行可以看到）：

```python
# 补充后的 tool_call_info：
tool_call_info = {
    'tool_name': 'optimize_structure',
    'tool_args': {
        # 原有的参数（来自 LLM 提取）
        'input_structure': 'https://bohrium.oss-cn-zhangjiakou.aliyuncs.com/.../Cu_bulk.cif',
        'model_path': 'https://bohrium.oss-cn-zhangjiakou.aliyuncs.com/.../dpa-2.4-7M.pt',
        'relax_cell': False,

        # 新增的参数（从 function_declarations 补充的默认值）
        'head': 'Omat24',              # 从默认值补充
        'force_tolerance': 0.01,        # 从默认值补充
        'max_iterations': 100           # 从默认值补充
    },
    'missing_tool_args': []  # 所有必需参数都有值，所有可选参数都有默认值
}
```

## 三、关键区别总结

| 特性 | `tool_args` | `function_declarations` |
|------|-------------|------------------------|
| **数据来源** | LLM 从用户描述中提取 | 工具的完整 Schema 定义 |
| **内容范围** | 只包含用户提到/LLM 推断的参数 | 包含工具的所有参数定义 |
| **参数值** | 包含用户指定的值或 LLM 推断的值 | 包含参数的类型、描述、默认值等元数据 |
| **完整性** | 不完整（只包含部分参数） | 完整（包含所有参数） |
| **用途** | 用于实际调用工具 | 用于补充缺失参数、验证参数、生成 JSON |

## 四、处理流程

```
用户输入："对这个结构进行几何优化"
  ↓
LLM 分析并生成 function_call
  ↓
提取 tool_args（只包含 LLM 认为需要的参数）
  {
    'input_structure': '...',
    'model_path': '...',
    'relax_cell': False
  }
  ↓
从 function_declarations 获取完整参数定义
  {
    'input_structure': {...},
    'model_path': {...},
    'head': {'default': 'Omat24'},      ← 有默认值
    'force_tolerance': {'default': 0.01}, ← 有默认值
    'max_iterations': {'default': 100},  ← 有默认值
    'relax_cell': {'default': False}
  }
  ↓
update_tool_call_info_with_function_declarations
  - 检查 tool_args 中缺少的参数
  - 如果有默认值，补充到 tool_args
  - 如果是必需参数但没有值，添加到 missing_tool_args
  ↓
最终的 tool_args（完整）
  {
    'input_structure': '...',
    'model_path': '...',
    'relax_cell': False,
    'head': 'Omat24',              ← 补充的默认值
    'force_tolerance': 0.01,       ← 补充的默认值
    'max_iterations': 100          ← 补充的默认值
  }
  ↓
生成 JSON 时，遍历 function_declarations 的所有参数
  - 确保所有参数都出现在 input_parameters 中
  - 如果参数在 tool_args 中，使用 tool_args 的值
  - 如果参数会被 edges 连接，value 设为 ""
  - 如果参数有默认值，使用默认值
  - 否则根据类型设置默认值
```

## 五、在 JSON 生成中的作用

### 为什么需要遍历 `function_declarations`？

在 `save_parameters_to_json` 中，我们遍历 `function_declarations` 的所有参数，而不是只使用 `tool_args`，原因如下：

1. **完整性**：确保 JSON 中包含工具的所有参数，即使某些参数没有被用户提到
2. **一致性**：所有节点的 JSON 格式一致，都包含完整的参数列表
3. **可维护性**：如果工具添加了新参数，JSON 会自动包含，无需修改代码

### 值的优先级（在 JSON 生成中）

```python
# 在 event_utils.py 的 save_parameters_to_json 中：
for param_name, param_schema in properties.items():
    # 优先级1：如果会被 edges 连接，value = ""
    if (step_index, param_name) in connected_params:
        param_value = ""

    # 优先级2：如果在 tool_args 中，使用 tool_args 的值
    elif param_name in tool_args:
        param_value = tool_args[param_name]

    # 优先级3：如果有默认值，使用默认值
    elif default_value is not None:
        param_value = default_value

    # 优先级4：根据类型设置默认值
    else:
        if param_type == 'str':
            param_value = ""
        elif param_type == 'int':
            param_value = 0
        # ...
```

## 六、实际例子对比

### 场景：用户说"对这个结构进行几何优化"

#### 初始 `tool_args`（LLM 提取）
```json
{
  "input_structure": "https://.../Cu_bulk.cif",
  "model_path": "https://.../dpa-2.4-7M.pt",
  "relax_cell": false
}
```
**只有 3 个参数**

#### `function_declarations`（完整定义）
```json
{
  "name": "optimize_structure",
  "parameters": {
    "properties": {
      "input_structure": {"type": "string", "description": "..."},
      "model_path": {"type": "string", "description": "..."},
      "head": {"type": "string", "default": "Omat24"},
      "force_tolerance": {"type": "number", "default": 0.01},
      "max_iterations": {"type": "integer", "default": 100},
      "relax_cell": {"type": "boolean", "default": false},
      "executor": {"type": "object"},
      "storage": {"type": "object"}
    }
  }
}
```
**包含 8 个参数**

#### 最终 JSON 的 `input_parameters`（遍历 function_declarations）
```json
{
  "input_parameters": [
    {
      "name": "input_structure",
      "type": "str",
      "value": "https://.../Cu_bulk.cif"  // 来自 tool_args
    },
    {
      "name": "model_path",
      "type": "str",
      "value": "https://.../dpa-2.4-7M.pt"  // 来自 tool_args
    },
    {
      "name": "head",
      "type": "str",
      "value": "Omat24"  // 来自 function_declarations 的默认值
    },
    {
      "name": "force_tolerance",
      "type": "float",
      "value": 0.01  // 来自 function_declarations 的默认值
    },
    {
      "name": "max_iterations",
      "type": "int",
      "value": 100  // 来自 function_declarations 的默认值
    },
    {
      "name": "relax_cell",
      "type": "bool",
      "value": false  // 来自 tool_args（用户/LLM 指定）
    }
    // 注意：executor 和 storage 被跳过（系统参数）
  ]
}
```
**包含 6 个参数（排除系统参数）**

## 七、关键要点

1. **`tool_args` 是不完整的**：只包含用户提到或 LLM 推断的参数
2. **`function_declarations` 是完整的**：包含工具的所有参数定义
3. **需要两者结合**：
   - 用 `tool_args` 获取用户指定的值
   - 用 `function_declarations` 补充默认值和缺失参数
   - 用 `function_declarations` 确保 JSON 包含所有参数

4. **在 JSON 生成中**：
   - 遍历 `function_declarations` 的所有参数（确保完整性）
   - 优先使用 `tool_args` 中的值（用户指定的值）
   - 其次使用 `function_declarations` 中的默认值
   - 如果参数会被 edges 连接，value 设为 `""`
