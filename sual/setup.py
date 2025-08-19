from setuptools import setup, find_packages

setup(
    name='sual',  # 包的名称，和要安装的包文件夹名称一般保持一致
    version='0.1',  # 包的版本号，可以自行设定和更新
    description='A package for some specific functionality (describe briefly)',  # 简要描述包的功能
    packages=find_packages(),  # 自动查找项目中的所有Python包，包括嵌套的子包
    install_requires=[],  # 这里可以列出该包依赖的其他Python库，如果有依赖的话
    # 以下可选，如果你的包中有非Python文件（如数据文件、配置文件等）需要包含进来，可以配置此项
    include_package_data=True,  
    # 以下可选，用于指定包的分类信息，便于在PyPI等平台分类查找，这里简单示例可以设置为空
    classifiers=[]  
)