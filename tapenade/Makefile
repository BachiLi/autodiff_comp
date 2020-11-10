bin/arap: image_arap.c image_arap_energy.c image_arap_energy_d.c stb_image.h adBuffer.h adStack.h
	mkdir -p bin
	gcc -g -Ofast -flto image_arap.c image_arap_energy.c image_arap_energy_d.c adBuffer.c adStack.c -o bin/arap

bin/xTAx: xTAx_eval.c adBuffer.h adStack.h
	mkdir -p bin
	gcc -g -Ofast -flto xTAx_eval.c adBuffer.c adStack.c -o bin/xTAx

bin/xTATAx: xTATAx_eval.c
	mkdir -p bin
	gcc -g -Ofast -flto xTATAx_eval.c -o bin/xTATAx