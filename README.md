# Jittor: a Just-in-time(JIT) deep learning framework

[Quickstart](#quickstart) | [Install](#install) | [Tutorial](#tutorial) | [Chinese](./README.cn.md)

Jittor is a high-performance deep learning framework based on JIT compiling and meta-operators. The whole framework and meta-operators are compiled just-in-time. A powerful op compiler and tuner are integrated into Jittor. It allowed us to generate high-performance code with specialized for your model.

The front-end language is Python. Module Design is used in the front-end, which is the most popular design for deeplearning framework interface. The back-end is implemented by high performance language, such as CUDA,C++.

The following example shows how to model a two-layer neural network step by step and train from scratch In a few lines of Python code.

# 框架

入口：

jittor/python/jittor/__init__.py  先创建（os.mknod 方法） 编译缓冲区cache_path 然后 上锁（fcntl 模块）

在 linux 环境下用 Python 进行项目开发过程中经常会遇到多个进程对同一个文件进行读写问题，而此时就要对文件进行加锁控制，在 Python 的 linux 版本下有个 fcntl 模块可以方便的对文件进行加、解锁控制。

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

# 创建缓存编译器 build cache_compile
cc_flags += pybind_include
cc_flags += f" -I{jittor_path}/src "
check_cache_compile()

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

