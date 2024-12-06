@echo off
if "%1" == "" (
  echo Usage: make.bat [target]
  exit /b 1
)

set SOURCEDIR=.
set BUILDDIR=_build

if "%1" == "html" (
  sphinx-build -b html %SOURCEDIR% %BUILDDIR%/html
  exit /b 0
)

if "%1" == "clean" (
  if exist %BUILDDIR% rmdir /s /q %BUILDDIR%
  exit /b 0
)

echo Unknown target: %1
exit /b 1
