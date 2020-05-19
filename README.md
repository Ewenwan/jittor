# Jittor: a Just-in-time(JIT) deep learning framework

[Quickstart](#quickstart) | [Install](#install) | [Tutorial](#tutorial) | [Chinese](./README.cn.md)

Jittor is a high-performance deep learning framework based on JIT compiling and meta-operators. The whole framework and meta-operators are compiled just-in-time. A powerful op compiler and tuner are integrated into Jittor. It allowed us to generate high-performance code with specialized for your model.

The front-end language is Python. Module Design is used in the front-end, which is the most popular design for deeplearning framework interface. The back-end is implemented by high performance language, such as CUDA,C++.

The following example shows how to model a two-layer neural network step by step and train from scratch In a few lines of Python code.

# 框架

入口：

jit_utils 编译、日志相关：

     jittor\src\utils\jit_utils.cc  PYBIND11_MODULE(jit_utils_core, m){}
     m.def("cache_compile", &jittor::jit_compiler::cache_compile); // 缓存编译
     pybind 调用 c++源码 实现
     jittor\src\utils\log.cc 日志 等 
     jittor\src\utils\cache_compile.cc 缓存编译
          找到编译命令行中的 输入文件input_names 输出文件名output_name  include包含文件extra 源文件include的头文件
          生成 编译 cache_key 
          调用 log.cc 中 执行编译命令 system_with_check()
          popen()函数通过创建一个管道，调用fork()产生一个子进程，执行一个shell以运行命令来开启一个进程。
 
 

import jittor的时候会运行__init__.py，再调用 compiler.py，然后调用shell把整个框架编译一遍...

jittor/python/jittor/__init__.py  先创建（os.mknod 方法） 编译缓冲区cache_path 然后 上锁（fcntl 模块）

jittor-master/python/jittor/lock.py

在 linux 环境下用 Python 进行项目开发过程中经常会遇到多个进程对同一个文件进行读写问题，而此时就要对文件进行加锁控制，在 Python 的 linux 版本下有个 fcntl 模块可以方便的对文件进行加、解锁控制。

lock_path = os.path.abspath(os.path.join(cache_path, "../jittor.lock"))

os.mknod(lock_path)

os.mknod() 方法用于创建一个指定文件名的文件系统节点（文件，设备特别文件或者命名pipe）。

```py
# 上锁
fcntl.flock(self.handle, fcntl.LOCK_EX) # 排他锁： 除加锁进程外其他进程没有对已加锁文件读写访问权限
# 解锁
fcntl.flock(self.handle, fcntl.LOCK_UN) # 解锁： 对加锁文件进行解锁
# fcntl.LOCK_SH 共享锁： 所有进程都没有写权限，即使加锁进程也没有，但所有进程都有读权限
# fcntl.LOCK_NB 非阻塞锁： 如果指定此参数，函数不能获得文件锁就立即返回，否则，函数会等待获得文件锁。

# LOCK_NB可以同LOCK_SH或LOCK_NB进行按位或（|）运算操作。
fcnt.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
```

jittor/python/jittor/compiler.py

pyjt_compiler.py 根据头文件编译生成 .c源文件 

```py
# 找到各种path
python_path = sys.executable
py3_config_path = sys.executable+"-config"
nvcc_path = env_or_try_find('nvcc_path', '/usr/local/cuda/bin/nvcc')
gdb_path = try_find_exe('gdb')
addr2line_path = try_find_exe('addr2line')
has_pybt = check_pybt(gdb_path, python_path)

# 编译标记 选项 flag
cc_flags += " -Wall -Werror -Wno-unknown-pragmas -std=c++14 -fPIC -march=native "
link_flags = " -lstdc++ -ldl -shared "

core_link_flags = ""
opt_flags = ""
kernel_opt_flags = os.environ.get("kernel_flags", "") + opt_flags + " -fopenmp "

if ' -O' not in cc_flags:
    opt_flags += " -O2 "
    kernel_opt_flags += " -Ofast "
lto_flags = ""
if os.environ.get("enable_lto") == "1":
    if cc_type == "icc":
        lto_flags = " -flto -ipo -ipo-c "
    elif cc_type == "g++":
        lto_flags = " -flto -fuse-linker-plugin "
    else:
        lto_flags = " -flto "

# pybind_include 路径
pybind_include = run_cmd(python_path+" -m pybind11 --includes")
extension_suffix = run_cmd(py3_config_path+" --extension-suffix")
 
# 创建 编译缓冲区

make_cache_dir(cache_path)
make_cache_dir(os.path.join(cache_path, "jit"))
make_cache_dir(os.path.join(cache_path, "obj_files"))
make_cache_dir(os.path.join(cache_path, "gen"))

# 1. 编译jit_utils_core 并注册为python模块  创建缓存编译器 build cache_compile
cc_flags += pybind_include
cc_flags += f" -I{jittor_path}/src "
check_cache_compile()  //  编译jit_utils_core 并注册为python模块

···


# 检查是否支持 check cuda
has_cuda = 0
check_cuda()



# 编译 jittor 
# build core
gen_jit_flags()
gen_jit_tests()
op_headers = run_cmd('find -L src/ops/ | grep "op.h$"', jittor_path).splitlines()
jit_src = gen_jit_op_maker(op_headers)

#  jittor 核心cc文件实现
at_beginning = [
    "src/ops/op_utils.cc",
    "src/event_queue.cc",
    "src/mem/allocator/sfrl_allocator.cc",
    "src/mem/allocator.cc",
]
at_last = [
    "src/profiler/profiler.cc",
    "src/executor.cc",
    "src/fetcher.cc",
]
compile(cc_path, cc_flags+opt_flags, files, 'jittor_core'+extension_suffix)


# TODO: move to compile_extern.py
compile_extern()  # 
# 多线程支持 setup_mpi()    # mpicc_path = env_or_try_find('mpicc_path', 'mpicc')
                 # mpi_src_dir = os.path.join(jittor_path, "extern", "mpi")
                 #  mpi = compile_custom_ops(mpi_src_files,  extra_flags=f" {mpi_flags} ", return_module=True, dlopen_flags=os.RTLD_GLOBAL | os.RTLD_NOW)
                 
# nvidia cuda支持  setup_nccl() 
    # url = "https://github.com/NVIDIA/nccl/archive/v2.6.4-1.tar.gz"
    #     nccl_include_path = os.environ.get("nccl_include_path")
    # nccl_lib_path = os.environ.get("nccl_lib_path")
    # nccl_path = os.path.join(str(Path.home()), ".cache", "jittor", "nccl")
    # nccl_ops = compile_custom_ops(nccl_src_files, 
    #    extra_flags=f" -I'{nccl_include_path}' {mpi_compile_flags} ")
    
    
# setup_cutt()
    # url = "https://github.com/Jittor/cutt/archive/master.zip"
    # cutt_include_path = os.environ.get("cutt_include_path")
    # cutt_lib_path = os.environ.get("cutt_lib_path")
    # cutt_path = os.path.join(str(Path.home()), ".cache", "jittor", "cutt")
    # cutt_op_dir = os.path.join(jittor_path, "extern", "cuda", "cutt", "ops")
    # cutt_op_files = [os.path.join(cutt_op_dir, name) for name in os.listdir(cutt_op_dir)]
    # cutt_ops = compile_custom_ops(cutt_op_files,  extra_flags=f" -I'{cutt_include_path}'")
    
# setup_mkl()
    # url = "https://github.com/intel/mkl-dnn/releases/download/v1.0.2/mkldnn_lnx_1.0.2_cpu_gomp.tgz"
    # mkl_include_path = os.environ.get("mkl_include_path")
    # mkl_lib_path = os.environ.get("mkl_lib_path")
    # mkl_path = os.path.join(str(Path.home()), ".cache", "jittor", "mkl")
    # mkl_lib_name = os.path.join(mkl_lib_path, "libmkldnn.so")

    # mkl_op_dir = os.path.join(jittor_path, "extern", "mkl", "ops")
    # mkl_op_files = [os.path.join(mkl_op_dir, name) for name in os.listdir(mkl_op_dir)]
    # mkl_ops = compile_custom_ops(mkl_op_files, 
    #     extra_flags=f" -I'{mkl_include_path}' -L'{mkl_lib_path}' -lmkldnn -Wl,-rpath='{mkl_lib_path}' ")
    
# setup_cuda_extern()


```
jittor设计了一些基础算子，应该是打算把所有的常见Op拆分成这些基础算子，然后做一些fusion。

scripting是通过把Python的源代码解析成语法树，然后转化成C++可执行代码来实现的。

[TorchScript 如何实现Python -> C++ 代码转换](https://zhuanlan.zhihu.com/p/136585481)

[用C给python写module的步骤(入门)](https://blog.csdn.net/xiaozoom/article/details/83097136)

就是用C语言编写核心代码，然后封装成PYTHON可以使用的形式，是比较常用的手段之一

需要文件如下：

file1 - 包含C代码，以及封装样板代码的C文件

必须 #include “Python.h”
, //位置-如果装了anaconda则在 anaconda3/include里
, //默认在/usr/local/include/python3.x/中
```c
用C定义好函数–比如一个阶乘函数,如果看懂这个函数都有难度的话…
int fac(int n){
if (n<2)
return 1;
else
return (n)*fac(n-1);
}
int main(){
…//中间部分为测试C函数的代码
}
然后是样板函数，样板代码有几个·组成部分：

//1.每个定义的函数都需要一个样板函数
static PyObject* 模块名_函数名(PyObject self,PyObject args){
// static PyObject * Extest_fac(PyObject *self,PyObject *args){
// 1.1 用 PyArg_ParseTuple 函数 解读 参数    python类型参数转出c参数
char * command;
// int num：
// PyArg_ParseTuple(args,"i",&num)
//PyArg_ParseTuple(args,“s”,&command);  // "s" 表示字符串变量  "i"表示int整形变量
//这句话意思为将args解读成char类型存入command位置中
// 1.2 调用c函数
int res;
res = fac(command);//假设函数名叫fac

//1.3 用Py_BuildValue来将 结果c变量转换 成 python变量 返回
//Py_BuildValue() returns a tuple
PyObject retval;//定义返回变量
//retval = (PyObject)Py_BuildValue(类型,C变量1,C变量2）
// return (PyObject*)Py_BuildValue("i",res);

return retval;
}

//2.方法定义-就是该module包含了哪些Methods
static PyMethodDef ExtestMethods[] = { //Extest 为模块名
{“fac”,Extest_fac,METH_VARARGS},
/* python中使用的名称，对应样板函数名称,METH_VARARGS 的意思是期待PYTHON-LEVEL的 参数*/
{“doppel”,Extest_doppel,METH_VARARGS},
{NULL,NULL} //表示函数信息结束
};

//3.模块定义
static struct PyModuleDef extestmodule = {
PyModuleDef_HEAD_INIT,
“Extest”, //名称
NULL, //doc
-1, //不懂
ExtestMethods //方法定义
};

//4.启动样板函数
PyMODINIT_FUNC
PyInit_Extest(void){ // Extest为模块名称
PyObject *m;
m = PyModule_Create(&extestmodule); //为模块定义
if(m==NULL)
return NULL;

/可以接触发异常等等/
return m;
}


```
file2 - setup.py文件，用于编译以及安装
```py
from distutils.core import setup,Extension

MOD = ‘Extest’ //名称
setup(name=MOD,ext_modules=[Extension(MOD,sources=[‘Extest1.c’])])//源代码·位置
```
#命令

python setup.py build

python setup.py install

jittor 算子 编译注册 python 模块

src\pyjt\py_obj_holder.h   包含 #include <Python.h>  按照python接口编写函数 可import到python环境中

```c
#define PYJF_MODULE_INIT(name) \                   // 初始化模块 name
PyMODINIT_FUNC PyInit_##name() { \
    PyObject *m; \                   // PyObject* 可以表示任意Python对象的封装数据类型
    try { \
        PyModuleDef *def = new PyModuleDef(); \    // 模块定义
        memset(def, 0, sizeof(PyModuleDef)); \
        def->m_name = #name; \                     // 模块名
        def->m_doc = ""; \                         // 模块说明
        def->m_size = -1; \
        Py_INCREF(def); \                          // increase  ref   增加引用计数
                                                   // 当你需要保护这个变量不被释放时才使用INCREF
        jittor::PyObjHolder holder(m = PyModule_Create(def)); \
                                                   // PyObjHolder 保存 PyObject 对象
                                                   // 析构对象 调用 Py_DECREF 减少引用计数
        init_module(def, m); \                     // 初始化模块定义 的 doc 
        holder.release(); \
    } catch(const std::exception& e) { \
        PyErr_SetString(PyExc_RuntimeError, e.what()); \
        return nullptr; \
    } \
    return m; \
}

static void init_module(PyModuleDef* mdef, PyObject* m) {
    mdef->m_doc = "Inner c++ core of jittor"; // 初始化模块定义 的 doc 
    jittor::init();       // 调用  op_registe
    //  op_registe({"number", "", "", {{&typeid(&make_number), (void*)&make_number}}});
    jittor::pyjt_def_all(m);
}

```

jittor\src\ops\op_register.cc  op_registe()   op_info 注册op unordered_map<string, OpInfo> op_info_map;   包含名字  和 函数指针构造器 
 
```c
struct OpInfo {
    string name, source_path, extra_flags;   // op算子名称 源码路径
    vector<pair<const std::type_info*, void*>> constructors;
        // 构造函数容器 类型:函数指针对 数组
    // string: var member name, uint64: var member offset
    vector<pair<string, uint64>> var_members;   // 变量数组

    template<class To, class ...Ts> auto get_constructor() {
        typedef To (*func_t)(Ts...);
        const auto& tid = typeid(func_t);  // 函数类型
        for (uint i=0; i<constructors.size(); i++)
            if (std::type_index(*(constructors[i].first)) == std::type_index(tid))
                return func_t(constructors[i].second);
        LOGf << "constructor" << name << tid.name() << "not found.";
        return func_t(nullptr);
    }
};

unordered_map<string, OpInfo> op_info_map;

void op_registe(const OpInfo& op_info) {
    ASSERT(!has_op(op_info.name)) << "Op" << op_info.name << "is already registed, "
        << "source_path:" << op_info.source_path << "extra_flags" << op_info.extra_flags;
    LOGvv << "registe op" << op_info.name
        << "\nsource_path:" << op_info.source_path
        << "\nextra_flags:" << op_info.extra_flags
        << "\nconstructors:" << op_info.constructors
        << "\nvar_members:" << op_info.var_members;
    op_info_map[op_info.name] = op_info;    // 添加该op算子 到 算子信息库中
}


```

jittor\src\op.cc   算子操作

```c

// op算子名转换

// convert xxx.yyy -> xxx
string Op::op_name_to_file_name(const string& s) {
    auto pos = s.find('.');
    return pos == string::npos ? s : s.substr(0, pos);
}

// convert xxx_xxx -> XxxXxx
string Op::file_name_to_class_name(const string& s) {
    char prev = '_';
    string res;
    res.reserve(s.size());
    for (char c : s) {
        if (c != '_') {
            if (prev == '_')
                res += c-'a'+'A';
            else
                res += c;
        }
        prev = c;
    }
    return res;
}

// 算子 运行
void Op::jit_run() {
    const char* jit_key = jk.to_cstring();
    auto iter = jit_ops.find(jit_key);
    if (iter != jit_ops.end()) {
        LOGvvv <<  "Jit op key found:" << jit_key << "jit op entry:" << (void*)iter->second;
        
        Profiler::record_and_run(iter->second, this, jit_key);
        // 算子仓库中有 该算子  直接运行 并记录 耗时
        return;
    }
    LOGvv << "Jit op key not found:" << jit_key;
    // compile JIT op
    string prev_jit_key = jit_key;
    auto op_entry = OpCompiler::do_compile(this);  // 如果没有则即时编译 
    string new_jit_key = get_jit_key();
    jit_ops[new_jit_key] = jit_ops[prev_jit_key] = op_entry; // 记录到算子库
    jit_key_mapper[prev_jit_key] = new_jit_key;
    LOGvv << "Get jit op entry:" << (void*)op_entry;
    
    Profiler::record_and_run(op_entry, this, new_jit_key.c_str()); // 运行 并记录 耗时
}


// 耗时记录 profile

for (int64_t i=0; i<rerun; i++) {
  auto start = std::chrono::high_resolution_clock::now();
  jit_entry(op);
  #ifdef HAS_CUDA
  if (use_cuda)
      checkCudaErrors(cudaDeviceSynchronize());
  #endif
  auto finish = std::chrono::high_resolution_clock::now();
  auto total_ns =  (int64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count();
  // 24ns function call overhead
  total_ns = std::max((int64_t)1, total_ns-24);
  iter->second.update(loop, total_ns, in, out, compute);
  LOGvvvv << "Duration" << total_ns >> "ns running" << op;
}

```



jittor/src/executor.cc 

整个编译流程大概是：

0.0 python代码转成 节点图??
   
     jittor/src/var_holder.cc   
       void sync_all(bool device_sync)/ sync(const vector<VarHolder*>& vh, bool device_sync) 
       调用 extern Executor exe;   exe.run_sync();
       
       jittor\python\jittor\__init__.py   jittor_exit() 调用 core.sync_all(True)
       jittor\src\var_holder.cc           fetch_sync()  调用 sync(True)
       jittor\src\fetcher.cc              fetch()       调用 sync(True)
       
       jittor\src\pybind\core.cc   

0.5 图优化（图遍历、算子融合、并查集）[code](https://github.com/Ewenwan/jittor/blob/04644cd7583f6ef4780685e2c9c4722962f1ea4e/src/executor.cc#L104)

1. 内存分配，执行到第i个op的时候，会把这个op的输出var所需要的内存先申请好，相关代码：allocator.cc, sfrl_allocator.cc
[code](https://github.com/Ewenwan/jittor/blob/04644cd7583f6ef4780685e2c9c4722962f1ea4e/src/executor.cc#L337)

```c
for (auto* var : op->outputs())
    var->alloc(allocator);
```

2. 算子key生成，申请好内存以后，就开始准备计算了，计算前会先调用 op:do_jit_prepare ( jit_prepare() 那个算子重写 算子参数定义) 来生成 op 的 jitkey，这个jitkey就是op的身份证，看一下这个op是不是已经被编译过了，相关代码：jit_key.cc parse_jit_keys()
[code](https://github.com/Ewenwan/jittor/blob/04644cd7583f6ef4780685e2c9c4722962f1ea4e/src/executor.cc#L339)

```c

op->do_prepare();
```

3. 算子编译，如果没有编译过，开始编译，相关代码：op_compiler.cc
[]()

```c
op->do_run_after_prepare();

    const char* jit_key = jk.to_cstring();
    auto iter = jit_ops.find(jit_key);
    if (iter != jit_ops.end()) {
        LOGvvv <<  "Jit op key found:" << jit_key << "jit op entry:" << (void*)iter->second;
        Profiler::record_and_run(iter->second, this, jit_key);
        return;
    }
    
// compile JIT op

    string prev_jit_key = jit_key;
    auto op_entry = OpCompiler::do_compile(this);
    string new_jit_key = get_jit_key();
    jit_ops[new_jit_key] = jit_ops[prev_jit_key] = op_entry;
    jit_key_mapper[prev_jit_key] = new_jit_key;
    LOGvv << "Get jit op entry:" << (void*)op_entry;
    Profiler::record_and_run(op_entry, this, new_jit_key.c_str());
```


4. 算子运行，如果编译过，开始执行，准备完成以后，检查一个op是否被编译过：Op::jit_run
[code](https://github.com/Ewenwan/jittor/blob/04644cd7583f6ef4780685e2c9c4722962f1ea4e/src/executor.cc#L365)

```c

void Op::do_run_after_prepare() {

    if (!jk.empty())
        jit_run();

    else
        run();
}
```


# 示例
```python
import jittor as jt
from jittor import Module
from jittor import nn
class Model(Module):
    def __init__(self):
        self.layer1 = nn.Linear(1, 10)
        self.relu = nn.Relu() 
        self.layer2 = nn.Linear(10, 1)
    def execute (self,x) :
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

def get_data(n): # generate random data for training test.
    for i in range(n):
        x = np.random.rand(batch_size, 1)
        y = x*x
        yield jt.float32(x), jt.float32(y)

model = Model()
learning_rate = 0.1
optim = nn.SGD(model.parameters(), learning_rate)

for i,(x,y) in enumerate(get_data(n)):
    pred_y = model(x)
    loss = ((pred_y - y)**2)
    loss_mean = loss.mean()
    optim.step(loss_mean)
    print(f"step {i}, loss = {loss_mean.data.sum()}")
```

## Contents

* [Quickstart](#quickstart)
* [Install](#install)
* [Tutorial](#tutorial)
* [Contributing](#contributing)
* [The Team](#theteam)
* [License](#license)



## Quickstart


We provide some jupyter notebooks to help you quick start with Jittor.


- [Example: Model definition and training][1]
- [Basics: Op, Var][2]
- [Meta-operator: Implement your own convolution with Meta-operator][3]

## Install


Jittor is written in Python and C++. It requires a compiler for JIT compilation, Currently, we support four compilers:


* CPU compiler (require at least one of the following)
    * g++ (>=5.4.0)
    * clang (>=8.0) recommend
* GPU compiler (optional)
    * nvcc (>=10.0)



Jittor environment requirements:

* System: Ubuntu >= 16.04
* Python version >= 3.7
* C++ compiler(g++ or clang)

Jittor offers three ways to install: pip, script or manual.



## Pip install


```bash
sudo apt install python3.7-dev libomp-dev
sudo python3.7 -m pip install git+https://github.com/Jittor/jittor.git
# if you cannot access github, please download code from our website:
#     wget https://cg.cs.tsinghua.edu.cn/jittor/assets/build/jittor.tgz
#     mkdir -p jittor && tar -xvf ./jittor.tgz -C jittor
#     sudo pip install ./jittor
python3.7 -m jittor.test.test_example
```


## single line script install


We provide single line command for quick installation the latest version of Jittor(Ubuntu>=16.04):


```bash
# install with clang and cuda
wget -O - https://raw.githubusercontent.com/Jittor/jittor/master/script/install.sh | with_clang=1 with_cuda=1 bash
# install with clang
wget -O - https://raw.githubusercontent.com/Jittor/jittor/master/script/install.sh | with_clang=1 bash
# install with g++ and cuda
wget -O - https://raw.githubusercontent.com/Jittor/jittor/master/script/install.sh | with_gcc=1 with_cuda=1 bash
# install with g++
wget -O - https://raw.githubusercontent.com/Jittor/jittor/master/script/install.sh | with_gcc=1 bash
```
After execution, the script will show some environment variables you need to export.


If you use Jittor for CPU computing, we strongly recommend clang(>=8.0) as the back-end compiler of Jittor. Because some customized optimizations will be enabled.



## manual install

We will show how to install Jittor in Ubuntu 16.04 step by step, Other Linux distributions may have similar commands.


### Step 1: Choose your back-end compiler


```bash
# g++
sudo apt install g++ build-essential libomp-dev

# OR clang++-8
wget -O - https://apt.llvm.org/llvm.sh > /tmp/llvm.sh
bash /tmp/llvm.sh 8
```
### Step 2: Install Python and python-dev


Jittor need python version >= 3.7.


```bash
sudo apt install python3.7 python3.7-dev
```

### Step 3: Run Jittor


The whole framework is compiled Just-in-time. Let's install jittor via pip


```bash
git clone https://github.com/Jittor/jittor.git
sudo pip3.7 install ./jittor
export cc_path="clang++-8"
# if other compiler is used, change cc_path
# export cc_path="g++"
# export cc_path="icc"

# run a simple test
python3.7 -m jittor.test.test_example
```
if the test is passed, your Jittor is ready.


### Optional Step 4: Enable CUDA


Using CUDA in Jittor is very simple, Just setup environment value `nvcc_path`


```bash
# replace this var with your nvcc location 
export nvcc_path="/usr/local/cuda/bin/nvcc" 
# run a simple cuda test
python3.7 -m jittor.test.test_cuda 
```
if the test is passed, your can use Jittor with CUDA by setting `use_cuda` flag.


```python
import jittor as jt
jt.flags.use_cuda = 1
```

### Optional Step 5: Run full tests


To check the integrity of Jittor, you can run full tests.


```bash
python3.7 -m jittor.test -v
```
if those tests are failed, please report bugs for us, and feel free to contribute ^_^


## Tutorial


In the tutorial section, we will briefly explain the basic concept of Jittor.


To train your model with Jittor, there are only three main concepts you need to know:


* Var: basic data type of jittor
* Operations: Jittor'op is simular with numpy

### Var


First, let's get started with Var. Var is the basic data type of jittor. Computation process in Jittor is asynchronous for optimization. If you want to access the data, `Var.data` can be used for synchronous data accessing.


```python
import jittor as jt
a = jt.float32([1,2,3])
print (a)
print (a.data)
# Output: float32[3,]
# Output: [ 1. 2. 3.]
```

And we can give the variable a name.


```python
c.name('c')
print(c.name())
# Output: c
```

###Operations


Jittor'op is simular with numpy. Let's try some operations. We create Var `a` and `b` via operation `jt.float32`, and add them. Printing those variables shows they have the same shape and dtype.


```python
import jittor as jt
a = jt.float32([1,2,3])
b = jt.float32([4,5,6])
c = a*b
print(a,b,c)
print(type(a), type(b), type(c))
# Output: float32[3,] float32[3,] float32[3,]
# Output: <class 'jittor_core.Var'> <class 'jittor_core.Var'> <class 'jittor_core.Var'>
```
Beside that, All the operators we used `jt.xxx(Var, ...)` have alias `Var.xxx(...)`. For example:


```python
c.max() # alias of jt.max(a)
c.add(a) # alias of jt.add(c, a)
c.min(keepdims=True) # alias of jt.min(c, keepdims=True)
```

if you want to know all the operation which Jittor supports. try `help(jt.ops)`. All the operation you found in `jt.ops.xxx`, can be used via alias `jt.xxx`.


```python
help(jt.ops)
# Output:
#   abs(x: core.Var) -> core.Var
#   add(x: core.Var, y: core.Var) -> core.Var
#   array(data: array) -> core.Var
#   binary(x: core.Var, y: core.Var, op: str) -> core.Var
#   ......
```
### More


If you want to know more about Jittor, please check out the notebooks below:


* Quickstart
    - [Example: Model definition and training][1]
    - [Basics: Op, Var][2]
    - [Meta-operator: Implement your own convolution with Meta-operator][3]
* Advanced
    - [Custom Op: write your operator with C++ and CUDA and JIT compile it][4]
    - [Profiler: Profiling your model][5]
    - Jtune: Tool for performance tuning



[1]: notebook/example.src.md	"example"
[2]: notebook/basics.src.md	"basics"
[3]: notebook/meta_op.src.md	"meta_op"
[4]: notebook/custom_op.src.md	"custom_op"
[5]: notebook/profiler.src.md	"profiler"

Those notebooks can be started in your own computer by `python3.7 -m jittor.notebook`


## Contributing


Jittor is still young. It may contain bugs and issues. Please report them in our bug track system. Contributions are welcome. Besides, if you have any ideas about Jittor, please let us know.




You can help Jittor in the following ways:

* Citing Jittor in your paper
* recommend Jittor to your friends
* Contributing code
* Contributed tutorials and documentation
* File an issue
* Answer jittor related questions
* Light up the stars
* Keep an eye on jittor
* ......

## Contact Us





Website: http://cg.cs.tsinghua.edu.cn/jittor/

Email: jittor@qq.com

File an issue: https://github.com/Jittor/jittor/issues

## The Team


Jittor is currently maintained by Dun Liang, Guo-Ye Yang, Guo-Wei Yang and Wen-Yang Zhou etc. from the [Tsinghua CSCG Group](https://cg.cs.tsinghua.edu.cn/). If you are also interested in Jittor and want to improve it, Please join us!


## License


Jittor is Apache 2.0 licensed, as found in the LICENSE.txt file.

