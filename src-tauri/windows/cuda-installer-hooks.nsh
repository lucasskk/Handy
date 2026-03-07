!include "LogicLib.nsh"

!macro HANDY_MARK_MISSING DLL_NAME
  ${If} $R0 == ""
    StrCpy $R0 "  - ${DLL_NAME}$\r$\n"
  ${Else}
    StrCpy $R0 "$R0  - ${DLL_NAME}$\r$\n"
  ${EndIf}
!macroend

!macro HANDY_CHECK_CUDA_DLL DLL_NAME
  StrCpy $1 0

  ReadEnvStr $2 "CUDA_PATH_V12_9"
  ${If} $2 != ""
    IfFileExists "$2\bin\${DLL_NAME}" 0 +2
      StrCpy $1 1
  ${EndIf}

  ${If} $1 == 0
    ReadEnvStr $2 "CUDA_PATH"
    ${If} $2 != ""
      IfFileExists "$2\bin\${DLL_NAME}" 0 +2
        StrCpy $1 1
    ${EndIf}
  ${EndIf}

  ${If} $1 == 0
    IfFileExists "$PROGRAMFILES\NVIDIA GPU Computing Toolkit\CUDA\*\bin\${DLL_NAME}" 0 +2
      StrCpy $1 1
  ${EndIf}

  ${If} $1 == 0
    !insertmacro HANDY_MARK_MISSING "${DLL_NAME}"
  ${EndIf}
!macroend

!macro HANDY_CHECK_CUDNN_DLL
  StrCpy $1 0

  IfFileExists "$PROGRAMFILES\NVIDIA\CUDNN\*\bin\*\x64\cudnn64_9.dll" 0 +2
    StrCpy $1 1

  ${If} $1 == 0
    !insertmacro HANDY_MARK_MISSING "cudnn64_9.dll"
  ${EndIf}
!macroend

!macro NSIS_HOOK_PREINSTALL
  StrCpy $R0 ""

  !insertmacro HANDY_CHECK_CUDNN_DLL
  !insertmacro HANDY_CHECK_CUDA_DLL "cublas64_12.dll"
  !insertmacro HANDY_CHECK_CUDA_DLL "cublasLt64_12.dll"
  !insertmacro HANDY_CHECK_CUDA_DLL "cudart64_12.dll"
  !insertmacro HANDY_CHECK_CUDA_DLL "cufft64_11.dll"

  ${If} $R0 != ""
    MessageBox MB_ICONSTOP|MB_TOPMOST|MB_OK "CUDA prerequisites are missing.$\r$\n$\r$\nHandy is configured for strict CUDA mode and cannot be installed until these runtime DLLs are available:$\r$\n$R0$\r$\n$\r$\nChecked: CUDA_PATH_V12_9, CUDA_PATH, %ProgramFiles%\\NVIDIA GPU Computing Toolkit\\CUDA\\*\\bin, and %ProgramFiles%\\NVIDIA\\CUDNN\\*\\bin\\*\\x64.$\r$\n$\r$\nInstall NVIDIA CUDA 12 and cuDNN 9 runtimes, then run the installer again."
    Abort
  ${EndIf}
!macroend
