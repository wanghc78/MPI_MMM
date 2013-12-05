################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
O_SRCS += \
../src/MMM1DColumn/MMM1DColumn.o 

C_SRCS += \
../src/MMM1DColumn/MMM1DColumn.c 

OBJS += \
./src/MMM1DColumn/MMM1DColumn.o 

C_DEPS += \
./src/MMM1DColumn/MMM1DColumn.d 


# Each subdirectory must supply rules for building sources it contributes
src/MMM1DColumn/%.o: ../src/MMM1DColumn/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cygwin C Compiler'
	mpicc -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


