
call Setup_Python_Paths.bat

call Config_Variables.bat

call echo https://github.com/anishathalye/neural-style


call set NEURAL_NETWORK_FILE="%DATASET_FOLDER%/StyleTransferNetwork/imagenet-vgg-verydeep-19.mat"

call set INPUT_IMAGE_NAME=StyleMe
call set INPUT_IMAGE_NAME=850_4223 
call set INPUT_IMAGE_NAME=DSC0328_50

call set STYLE_IMG_NAME=shen-zhou_mount-lu
call set STYLE_IMG_NAME=shen-zhou_landscape
call set STYLE_IMG_NAME=VanGogh-StarryNight-1024px

call set STYLE_FILES="%DATASET_FOLDER%/StyleTransferInputPhotos/%STYLE_IMG_NAME%.jpg"


call set INPUT_FILES="%DATA_FOLDER%/DataPreperation/ToBeStyled/%INPUT_IMAGE_NAME%.jpg"

call set OUTPUT_FILE_DIRECTORY=%DATA_TF_OUTPUT%%INPUT_IMAGE_NAME%-%STYLE_IMG_NAME%
call set OUTPUT_FILE="%OUTPUT_FILE_DIRECTORY%/%INPUT_IMAGE_NAME%.jpg"
call set OUTPUT_FILE_MULTIPLE=%OUTPUT_FILE_DIRECTORY%/%INPUT_IMAGE_NAME%_{:05}.jpg

call set PRESERVE_COLORS="--preserve-colors"
call set OVERWRITE="--overwrite"

call mkdir %OUTPUT_FILE_DIRECTORY%

call python ../Projects/neural-style-master/neural_style.py --content %INPUT_FILES% --styles %STYLE_FILES% --output %OUTPUT_FILE% --network %NEURAL_NETWORK_FILE%  --checkpoint-output %OUTPUT_FILE_MULTIPLE%  --checkpoint-iterations 50

rem  --content-weight=0.5 --learning-rate=1000 --style-blend-weights=0.5 --pooling="avg" %PRESERVE_COLORS% --iterations 2500


