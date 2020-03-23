
call Setup_Python_Paths.bat

call Config_Variables.bat

call cd %TF_OD_Path%

call python ../../../../Projects/wip/tensorflow_webcam_test.py

call cd %TF_OD_PATH_RETURN%


