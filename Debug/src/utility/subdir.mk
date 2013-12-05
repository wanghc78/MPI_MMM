################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
O_SRCS += \
../src/utility/utility.o 

C_SRCS += \
../src/utility/utility.c 

OBJS += \
./src/utility/utility.o 

C_DEPS += \
./src/utility/utility.d 


# Each subdirectory must supply rules for building sources it contributes
src/utility/%.o: ../src/utility/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cygwin C Compiler'
	mpicc -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


