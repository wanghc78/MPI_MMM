################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../src/MMM2D/MMM2D.c 

OBJS += \
./src/MMM2D/MMM2D.o 

C_DEPS += \
./src/MMM2D/MMM2D.d 


# Each subdirectory must supply rules for building sources it contributes
src/MMM2D/%.o: ../src/MMM2D/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: Cygwin C Compiler'
	mpicc -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


