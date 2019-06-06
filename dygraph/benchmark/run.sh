#!/bin/bash
# @author Divano 19-06-06

# define help
if ([ ${#@} -ne 0 ] && [ "${@#"--help"}" = "" ]) || ([ ${#@} -ne 0 ] && [ "${@#"-h"}" = "" ]); then
	echo "------ help -------"
	echo "guide: sh ./run.sh cardnumber"
	echo "example: sh ./run.sh 1"
	exit 0
fi

# parameter check
if [ ${#@} -eq 0 ]; then
	echo "ERROR: illegal input!"
	exit 0
fi

# def variable
workdir="$PWD"
resultfolder="$PWD/result"
currentworkfolder="$(date +%y-%m-%d)"
cardno=$1

# create folder
if [ -d $resultfolder ]; then
	echo "The results folder already exists!"
else
	mkdir $resultfolder
fi

if [ -d "$resultfolder/$currentworkfolder" ]; then
	echo "The work folder already exists!"
else
    mkdir "$resultfolder/$currentworkfolder"
fi

workfolder="$resultfolder/$currentworkfolder"


# run models function
# file 根目录到file文件路径
# cardno 第几张卡
# modelname 模型名字
# dir 模型路径
# 参数 dir, file, modelname
run_models(){
	# dir 设置是因为python相对路径问题
	dir=$1
	file=$2
	modelname=$3
	# test
	# echo $dir
	# echo $cardno
	# echo $modelname
	# exit 0
	cd $dir
	if [ -d $workfolder/$modelname ]; then
		echo "The ${modelname} folder already exists!"
	else
		mkdir $workfolder/$modelname
	fi
	echo "----${modelname} start----"
	export CUDA_VISIBLE_DEVICES=$cardno
	export FLAGS_fraction_of_gpu_memory_to_use=0.92
	echo "----run model benchmark----"
	python3.6 -u $file >  $workfolder/$modelname/$modelname.log 2>&1 
	echo $workfolder/$modelname/${modelname}_mem.log
	echo "----model benchmark done!----"
	echo "----run mem benchmark----"
	nvidia-smi -lms 250 --query-gpu=memory.total,memory.used,memory.free,index,timestamp,name --format=csv -i $cardno > $workfolder/$modelname/${modelname}_mem.log &
	export FLAGS_fraction_of_gpu_memory_to_use=0.0
	# echo "FLAGS_fraction_of_gpu_memory_to_use:"$FLAGS_fraction_of_gpu_memory_to_use
	export CUDA_VISIBLE_DEVICES=$cardno
	# echo "CUDA_VISIBLE_DEVICES:"$CUDA_VISIBLE_DEVICES
	python3.6 -u $file
	# wait gpu count exit
	sleep 2
	# kill gpu count 
	kill -9 `ps -ef | grep "nvidia-smi -lms 250" | grep -v grep | awk '{print $2}'`
	echo "----mem benchmark done!----"
	echo "----${modelname} done!----"
	cd $workdir
}

# -------- 执行函数 ----------#
# -------- 添加模型修改此处即可 ---------#
# run models
# mnist
# run_models "./models" "./dygraph_mnist.py" "mnist"
# run_models "./models" "./reinforce.py" "reinforce"
