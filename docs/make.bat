@ECHO OFF

pushd %~dp0

if "%SPHINXBUILD%" == "" (
    set SPHINXBUILD=sphinx-build
)

set SOURCEDIR= .
set BUILDDIR= _build

if "%1" == "" goto help

%SPHINXBUILD% -b %1 %SOURCEDIR% %BUILDDIR%/%1

goto end

:help
echo.Please use `make.bat html` to build the documentation.
:end

popd
