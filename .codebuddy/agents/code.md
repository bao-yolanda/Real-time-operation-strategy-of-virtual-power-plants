---
name: code
description: 
tools: list_files, search_file, search_content, read_file, read_lints, replace_in_file, write_to_file, execute_command, create_rule, delete_files, web_fetch, use_skill, web_search
agentMode: manual
enabled: true
enabledAutoRun: true
---
首先你的基础的数学模型都是文件proportional_control里面的matlab实现
主要希望你核对python实现（文件夹myexp）里面的实现的逻辑和matlab文件的数据处理和建模逻辑一直
但是可以根据两个语言的数据使用逻辑不通适当迁移
但是请保持数据的输入输出，使用 模型的建立是一样的，模型的有些功能可以根据我的指令修改，但是如果我没有给指令的地方就是按照matlab实现
请避免使用硬编码，可以根据参数设置调整