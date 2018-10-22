################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/ENHANCE/EnhanceImage_CPU.cpp \
../src/ENHANCE/clahe.cpp 

CU_SRCS += \
../src/ENHANCE/EnhanceImage.cu \
../src/ENHANCE/Filter.cu \
../src/ENHANCE/bilateral_kernel.cu 

CU_DEPS += \
./src/ENHANCE/EnhanceImage.d \
./src/ENHANCE/Filter.d \
./src/ENHANCE/bilateral_kernel.d 

OBJS += \
./src/ENHANCE/EnhanceImage.o \
./src/ENHANCE/EnhanceImage_CPU.o \
./src/ENHANCE/Filter.o \
./src/ENHANCE/bilateral_kernel.o \
./src/ENHANCE/clahe.o 

CPP_DEPS += \
./src/ENHANCE/EnhanceImage_CPU.d \
./src/ENHANCE/clahe.d 


# Each subdirectory must supply rules for building sources it contributes
src/ENHANCE/%.o: ../src/ENHANCE/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/usr/include/opencv -I/usr/include/GL -I../include -O3 -Xcompiler -fPIC -Xcompiler -fopenmp -ccbin aarch64-linux-gnu-g++ -gencode arch=compute_53,code=sm_53 -m64 -odir "src/ENHANCE" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I/usr/include/opencv -I/usr/include/GL -I../include -O3 -Xcompiler -fPIC -Xcompiler -fopenmp --compile --relocatable-device-code=false -gencode arch=compute_53,code=compute_53 -gencode arch=compute_53,code=sm_53 -m64 -ccbin aarch64-linux-gnu-g++  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/ENHANCE/%.o: ../src/ENHANCE/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/usr/include/opencv -I/usr/include/GL -I../include -O3 -Xcompiler -fPIC -Xcompiler -fopenmp -ccbin aarch64-linux-gnu-g++ -gencode arch=compute_53,code=sm_53 -m64 -odir "src/ENHANCE" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I/usr/include/opencv -I/usr/include/GL -I../include -O3 -Xcompiler -fPIC -Xcompiler -fopenmp --compile -m64 -ccbin aarch64-linux-gnu-g++  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


