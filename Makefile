# Makefiel for SRCNN git cloned mod.
# by Raphael Kim

CPP = gcc
CXX = g++
CUDA = nvcc
AR  = ar

OPENCV_INCS := `pkg-config --cflags opencv4`
OPENCV_LIBS := `pkg-config --libs opencv4`
FFMPEG_INCS := `pkg-config --cflags libavcodec libavformat libavutil libswscale`
FFMPEG_LIBS := `pkg-config --libs libavcodec libavformat libavutil libswscale`


SRC_PATH = src
OBJ_PATH = obj
BIN_PATH = bin
TARGET   = srcnn

SRCS += $(SRC_PATH)/frawscale.cpp
SRCS += $(SRC_PATH)/tick.cpp
# SRCS += $(SRC_PATH)/srcnn.cpp
SRCS += $(SRC_PATH)/convdata.cpp
SRCS += $(SRC_PATH)/video.cpp
OBJS = $(SRCS:$(SRC_PATH)/%.cpp=$(OBJ_PATH)/%.o)

SRCS_CUDA = $(SRC_PATH)/convdataCuda.cu
SRCS_CUDA += $(SRC_PATH)/srcnn.cu
OBJS_CUDA = $(SRCS_CUDA:$(SRC_PATH)/%.cu=$(OBJ_PATH)/%.o)

CFLAGS  =  -Xcompiler -fopenmp -rdc=true
CFLAGS += -I$(SRC_PATH)
CFLAGS += $(OPENCV_INCS)
CFLAGS += $(FFMPEG_INCS)

# Static build may require static-configured openCV.
LFLAGS  =
LFLAGS += $(OPENCV_LIBS)
LFLAGS += $(FFMPEG_LIBS)
LFLAGS += -Xcompiler -static-libgcc -Xcompiler -static-libstdc++
LFLAGS += -Xcompiler -s -Xcompiler -ffast-math -O3

all: prepare $(BIN_PATH)/$(TARGET)

prepare:
	@mkdir -p $(OBJ_PATH)
	@mkdir -p $(BIN_PATH)

clean:
	@rm -rf $(OBJ_PATH)/*.o
	@rm -rf $(BIN_PATH)/$(TARGET)

$(OBJS): $(OBJ_PATH)/%.o: $(SRC_PATH)/%.cpp
	@echo "Compiling $< ..."
	@$(CUDA) $(CFLAGS) -c $< -o $@

$(OBJS_CUDA): $(OBJ_PATH)/%.o: $(SRC_PATH)/%.cu
	@echo "Compiling $< ..."
	@$(CUDA) $(CFLAGS) -c $< -o $@

$(BIN_PATH)/$(TARGET): $(OBJS) $(OBJS_CUDA)
	@echo "Linking $@ ..."
	@$(CUDA) $(OBJ_PATH)/*.o $(CFLAGS) $(LFLAGS) -o $@
