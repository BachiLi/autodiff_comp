bin/arap: image_arap.c image_arap_energy.c image_arap_energy_d.c stb_image.h adBuffer.h adStack.h
	mkdir -p bin
	gcc -g -Ofast image_arap.c image_arap_energy.c image_arap_energy_d.c adBuffer.c adStack.c -o bin/arap

bin/xTAx: xTAx_eval.c adBuffer.h adStack.h
	mkdir -p bin
	gcc -g -Ofast xTAx_eval.c adBuffer.c adStack.c -o bin/xTAx