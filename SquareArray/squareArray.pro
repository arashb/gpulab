win32 {
  TEMPLATE = vcapp
  CONFIG += debug console
}
unix {
  TEMPLATE = app
  CONFIG += qt debug warn_on 
}

TARGET = squareArray
DEPENDPATH +=



#--- additional preprocessor directives ---
win32 {
  DEFINES += _CONSOLE
  DEFINES += _DEBUG
  DEFINES += NOMINMAX
}


#--- Input files ---

#CUDA_HEADERS += 
CUDA_SOURCES += squareArray.cu


########################################################################
#  CUDA
########################################################################
win32 {
 INCLUDEPATH += $(CUDA_INC_DIR)
 INCLUDEPATH += $(CUDA_SDK_INC_DIR)
 QMAKE_LIBDIR += $(CUDA_LIB_DIR)
 QMAKE_LIBDIR += $(CUDA_SDK_LIB_DIR)
 LIBS += -lcudart -lcuda

 cuda.output = $$OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.obj
 cuda.commands = $(CUDA_BIN_DIR)/nvcc.exe -c -Xcompiler $$join(QMAKE_CXXFLAGS,",") $$join(INCLUDEPATH,'" -I "','-I "','"') ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
 #cuda.commands = $(CUDA_BIN_DIR)/nvcc.exe -ccbin '"$(VCInstallDir)bin"' -c -D_DEBUG -DWIN32 -D_CONSOLE -Xcompiler $$join(QMAKE_CXXFLAGS,",") -I'"./ "' $$join(INCLUDEPATH,'" -I "','-I "','"') ${QMAKE_FILE_NAME} -o "$(ConfigurationName)"${QMAKE_FILE_OUT}
}

unix {
 # auto-detect CUDA path
 #CUDA_DIR = $$system(which nvcc | sed 's,/bin/nvcc$,,')
 # set CUDA path manually
 CUDA_DIR = $(CUDADIR)
 INCLUDEPATH += $$CUDA_DIR/include
 INCLUDEPATH += $(CUDA_SDK_INC_DIR)

 # Compiler flags tuned for my system
 #QMAKE_CXXFLAGS += -I../ -O99 -pipe -g -Wall
 QMAKE_LIBDIR += $$CUDA_LIB_DIR
 QMAKE_LIBDIR += $(CUDADRVDIR)
 LIBS += -L$(CUDA_LIB_DIR) -L$(CUDA_SDK_LIB_DIR) -lcuda -lcudart


 cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.obj
 cuda.commands = nvcc -c -Xcompiler $$join(QMAKE_CXXFLAGS,",") $$join(INCLUDEPATH," -I ",-I ",") ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
 #cuda.depends = nvcc -M -Xcompiler $$join(QMAKE_CXXFLAGS,",") $$join(INCLUDEPATH," -I ",-I ",") ${QMAKE_FILE_NAME} | sed "s,^.*: ,," | sed "s,^ *,," | tr -d '\\\n'
 cuda.dependcy_type = TYPE_C
 cuda.depend_command = nvcc -M -Xcompiler $$join(QMAKE_CXXFLAGS,",") $$join(INCLUDEPATH," -I ",-I ",") ${QMAKE_FILE_NAME} | sed "s,^.*: ,," | sed "s,^ *,," | tr -d '\\\n'
}
cuda.input = CUDA_SOURCES
QMAKE_EXTRA_UNIX_COMPILERS += cuda
########################################################################