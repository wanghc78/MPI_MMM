################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../src/MMM3D/MMM3D.c 

OBJS += \
./src/MMM3D/MMM3D.o 

C_DEPS += \
./src/MMM3D/MMM3D.d 


# Each subdirectory must supply rules for building sources it contributes
src/MMM3D/%.o: ../src/MMM3D/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cygwin C Compiler'
	mpicc -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


