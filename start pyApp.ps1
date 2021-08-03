
# paths
# $sysPath = 'C:\Users\BlaizeOlle\Source\Repos\PiVue\'
# $sysPath = 'C:\Users\blaiz\Source\Repos\PiVue\'
# $sysPath = 'C:\Users\ScottKOlle\source\repos\PiVue\'
$sysPath = 'C:\Users\daddy\source\repos\PiVue\'
Set-Location $sysPath
# C:\Users\daddy\source\repos\PiVue\webapp\runPy.py
# PiVue backent pyApp`
# # runPyApp.bat .\webapp\runpy.py
$runPath = '.\webapp\runPy.py'
Start-Process -FilePath "cmd" -ArgumentList "/c runPyApp.bat $runPath"

