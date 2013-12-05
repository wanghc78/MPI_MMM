################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../src/MMM1DRow/MMM1DRow.c 

OBJS += \
./src/MMM1DRow/MMM1DRow.o 

C_DEPS += \
./src/MMM1DRow/MMM1DRow.d 


# Each subdirectory must supply rules for building sources it contributes
src/MMM1DRow/%.o: ../src/MMM1DRow/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cygwin C Compiler'
	mpicc -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


