<?php
/**
 * Script para la generaci�n de CAPTCHAS
 *
 * @author  Jose Rodriguez <josecl@gmail.com>
 * @license GPLv3
 * @link    https://github.com/josecl/cool-php-captcha
 * @package captcha
 * @version 0.3.2
 *
 */

var_dump($_GET);
$dst_dir = './';

if (isset($_GET['dest-directory'])) {	
	if (!file_exists ( $_GET['dest-directory'] )) {
		$error =  "Directory ".$_GET['dest-directory']. " does not exist.\n";
		throw new Exception($error);
	}
	
	if (!is_writable ( $_GET['dest-directory'] )) {
		$error =  "Cannot access ".$_GET['dest-directory']. ". Permission denied.\n";
		throw new Exception($error);
	}
	
	$dst_dir = $_GET['dest-directory'];
}


if (isset($_GET['captext-list'])) {
	if (!file_exists ( $_GET['captext-list'] )) {
		$error =  "File ".$_GET['captext-list']. " does not exist.\n";
		throw new Exception($error);
	}
	
	if (!is_writable ( $_GET['captext-list'] )) {
		$error =  "Cannot access ".$_GET['captext-list']. ". Permission denied.\n";
		throw new Exception($error);
	}
	
	$captext_list = $_GET['captext-list'];
} else {
	$error =  "Please specify captcha text list files in captext-list=filename";
	throw new Exception($error);
}


session_start();
  
// read text line by line of the file
$handle = fopen($captext_list, "r");
if ($handle) {
    while (($line = fgets($handle)) !== false) {
        // new instance of captcha
        $captcha = new SimpleCaptcha();

		// OPTIONAL Change configuration...
		//$captcha->wordsFile = 'words/es.php';
		//$captcha->session_var = 'secretword';
		//$captcha->lineWidth = 3;
		//$captcha->scale = 3; $captcha->blur = true;
		//$captcha->resourcesPath = "/var/cool-php-captcha/resources";

		// OPTIONAL Simple autodetect language example
		/*
		if (!empty($_SERVER['HTTP_ACCEPT_LANGUAGE'])) {
			$langs = array('en', 'es');
			$lang  = substr($_SERVER['HTTP_ACCEPT_LANGUAGE'], 0, 2);
			if (in_array($lang, $langs)) {
				$captcha->wordsFile = "words/$lang.php";
			}
		}
		*/

		$captcha->imageFormat = 'png';
		$captcha->minWordLength = 6;
		$captcha->maxWordLength = 6;
		$captcha->dst_dir = $dst_dir;

		// Option for create data samples
		$captcha->fonts = array(
				'Antykwa'  => array('spacing' => -3, 'minSize' => 27, 'maxSize' => 30, 'font' => 'AntykwaBold.ttf'),        
			);

		$captcha->colors = array(
			array(0,0,0), // black    
		  );        
        
        // remove special char
        $static_text = preg_replace('/[^A-Za-z0-9\-]/', '', $line);;
        
        // Image generation
		$captcha->CreateImage($static_text=$static_text);
    }

    fclose($handle);
} else {
    // error opening the file.
    $error =  "Error opening file\n";
	throw new Exception($error);
}






























/**
 * SimpleCaptcha class
 *
 */
class SimpleCaptcha {

    /** 
     * Difficulty level (if used, keep 'Wave configuracion' on default)
     * normal = 1 
     * closer to 0 is more easy, for ex: 0.4
     * closer to 2 is more hard, for ex: 1.8
     * smaller then 0 and bigger than 2 is caped to min or max
     */
    public $difficulty = 1;

    /** Width of the image */
    public $width  = 200;

    /** Height of the image */
    public $height = 70;

    /** Dictionary word file (empty for random text) */
    public $wordsFile = 'words/en.php';

    /**
     * Path for resource files (fonts, words, etc.)
     *
     * "resources" by default. For security reasons, is better move this
     * directory to another location outise the web server
     *
     */
    public $resourcesPath = 'coolcaptcha-resource';
    
    public $dst_dir = './';

    /** Min word length (for non-dictionary random text generation) */
    public $minWordLength = 5;

    /**
     * Max word length (for non-dictionary random text generation)
     * 
     * Used for dictionary words indicating the word-length
     * for font-size modification purposes
     */
    public $maxWordLength = 8;

    /** Sessionname to store the original text */
    public $session_var = 'captcha';

    /** Background color in RGB-array */
    public $backgroundColor = array(255, 255, 255);

    /** Foreground colors in RGB-array */
    public $colors = array(
        array(27,78,181), // blue
        array(22,163,35), // green
        array(214,36,7),  // red
    );

    /** Shadow color in RGB-array or null */
    public $shadowColor = null; //array(0, 0, 0);

    /** Horizontal line through the text */
    public $lineWidth = 0;

    /**
     * Font configuration
     *
     * - font: TTF file
     * - spacing: relative pixel space between character
     * - minSize: min font size
     * - maxSize: max font size
     */
    public $fonts = array(
        'Antykwa'  => array('spacing' => -3, 'minSize' => 27, 'maxSize' => 30, 'font' => 'AntykwaBold.ttf'),
        'Candice'  => array('spacing' =>-1.5,'minSize' => 28, 'maxSize' => 31, 'font' => 'Candice.ttf'),
        'DingDong' => array('spacing' => -2, 'minSize' => 24, 'maxSize' => 30, 'font' => 'Ding-DongDaddyO.ttf'),
        'Duality'  => array('spacing' => -2, 'minSize' => 30, 'maxSize' => 38, 'font' => 'Duality.ttf'),
        'Heineken' => array('spacing' => -2, 'minSize' => 24, 'maxSize' => 34, 'font' => 'Heineken.ttf'),
        'Jura'     => array('spacing' => -2, 'minSize' => 28, 'maxSize' => 32, 'font' => 'Jura.ttf'),
        'StayPuft' => array('spacing' =>-1.5,'minSize' => 28, 'maxSize' => 32, 'font' => 'StayPuft.ttf'),
        'Times'    => array('spacing' => -2, 'minSize' => 28, 'maxSize' => 34, 'font' => 'TimesNewRomanBold.ttf'),
        'VeraSans' => array('spacing' => -1, 'minSize' => 20, 'maxSize' => 28, 'font' => 'VeraSansBold.ttf'),
    );

    /** Wave configuracion in X and Y axes */
    public $Yperiod    = 12;
    public $Yamplitude = 14;
    public $Xperiod    = 11;
    public $Xamplitude = 5;

    /** letter rotation clockwise */
    public $maxRotation = 8;

    /**
     * Internal image size factor (for better image quality)
     * 1: low, 2: medium, 3: high
     */
    public $scale = 2;

    /** 
     * Blur effect for better image quality (but slower image processing).
     * Better image results with scale=3
     */
    public $blur = false;

    /** Debug? */
    public $debug = false;
    
    /** Image format: jpeg or png */
    public $imageFormat = 'jpeg';


    /** GD image */
    public $im;










    public function __construct($config = array()) {
    }







    public function CreateImage($static_text=null) {
        $ini = microtime(true);

        // cap difficulty
        if($this->difficulty >2) $this->difficulty = 2;
        if($this->difficulty<=0) $this->difficulty = 0.1;

        /** Initialization */
        $this->ImageAllocate();
        
        /** Text insertion */
        if ($static_text == null)
			$text = $this->GetCaptchaText();
		else $text = $static_text;
		
        $fontcfg  = $this->fonts[array_rand($this->fonts)];
        $this->WriteText($text, $fontcfg);

        $_SESSION[$this->session_var] = $text;

        /** Transformations */
        if (!empty($this->lineWidth)) {
            $this->WriteLine();
        }
        $this->WaveImage();
        if ($this->blur && function_exists('imagefilter')) {
            imagefilter($this->im, IMG_FILTER_GAUSSIAN_BLUR);
        }
        $this->ReduceImage();


        if ($this->debug) {
            imagestring($this->im, 1, 1, $this->height-8,
                "$text {$fontcfg['font']} ".round((microtime(true)-$ini)*1000)."ms",
                $this->GdFgColor
            );
        }


        /** Output */
        $this->WriteImage($text);
        $this->Cleanup();
    }









    /**
     * Creates the image resources
     */
    protected function ImageAllocate() {
        // Cleanup
        if (!empty($this->im)) {
            imagedestroy($this->im);
        }

        $this->im = imagecreatetruecolor($this->width*$this->scale, $this->height*$this->scale);

        // Background color
        $this->GdBgColor = imagecolorallocate($this->im,
            $this->backgroundColor[0],
            $this->backgroundColor[1],
            $this->backgroundColor[2]
        );
        imagefilledrectangle($this->im, 0, 0, $this->width*$this->scale, $this->height*$this->scale, $this->GdBgColor);

        // Foreground color
        $color           = $this->colors[mt_rand(0, sizeof($this->colors)-1)];
        $this->GdFgColor = imagecolorallocate($this->im, $color[0], $color[1], $color[2]);

        // Shadow color
        if (!empty($this->shadowColor) && is_array($this->shadowColor) && sizeof($this->shadowColor) >= 3) {
            $this->GdShadowColor = imagecolorallocate($this->im,
                $this->shadowColor[0],
                $this->shadowColor[1],
                $this->shadowColor[2]
            );
        }
    }





    /**
     * Text generation
     *
     * @return string Text
     */
    protected function GetCaptchaText() {
        $text = $this->GetDictionaryCaptchaText();
        if (!$text) {
            $text = $this->GetRandomCaptchaText();
        }
        return $text;
    }






    /**
     * Random text generation
     *
     * @return string Text
     */
    protected function GetRandomCaptchaText($length = null) {
        if (empty($length)) {
            $length = rand($this->minWordLength, $this->maxWordLength);
        }

        $words  = "abcdefghijlmnopqrstvwyz";
        $vocals = "aeiou";

        $text  = "";
        $vocal = rand(0, 1);
        for ($i=0; $i<$length; $i++) {
            if ($vocal) {
                $text .= substr($vocals, mt_rand(0, 4), 1);
            } else {
                $text .= substr($words, mt_rand(0, 22), 1);
            }
            $vocal = !$vocal;
        }
        return $text;
    }









    /**
     * Random dictionary word generation
     *
     * @param boolean $extended Add extended "fake" words
     * @return string Word
     */
    function GetDictionaryCaptchaText($extended = false) {
        if (empty($this->wordsFile)) {
            return false;
        }

        // Full path of words file
        if (substr($this->wordsFile, 0, 1) == '/') {
            $wordsfile = $this->wordsFile;
        } else {
            $wordsfile = $this->resourcesPath.'/'.$this->wordsFile;
        }

        if (!file_exists($wordsfile)) {
            return false;
        }

        $fp     = fopen($wordsfile, "r");
        $length = strlen(fgets($fp));
        if (!$length) {
            return false;
        }
        $line   = rand(1, (filesize($wordsfile)/$length)-2);
        if (fseek($fp, $length*$line) == -1) {
            return false;
        }
        $text = trim(fgets($fp));
        fclose($fp);


        /** Change ramdom volcals */
        if ($extended) {
            $text   = preg_split('//', $text, -1, PREG_SPLIT_NO_EMPTY);
            $vocals = array('a', 'e', 'i', 'o', 'u');
            
            foreach ($text as $i => $char) {
                if (mt_rand(0, 1) && in_array($char, $vocals)) {
                    $text[$i] = $vocals[mt_rand(0, 4)];
                }
            }
            $text = implode('', $text);
        }

        return $text;
    }








    /**
     * Horizontal line insertion
     */
    protected function WriteLine() {

        $x1 = $this->width*$this->scale*.15;
        $x2 = $this->textFinalX;
        $y1 = rand($this->height*$this->scale*.40, $this->height*$this->scale*.65);
        $y2 = rand($this->height*$this->scale*.40, $this->height*$this->scale*.65);
        $width = $this->lineWidth/2*$this->scale;

        for ($i = $width*-1; $i <= $width; $i++) {
            imageline($this->im, $x1, $y1+$i, $x2, $y2+$i, $this->GdFgColor);
        }
    }




    /**
     * Text insertion
     */
    protected function WriteText($text, $fontcfg = array()) {
        if (empty($fontcfg)) {
            // Select the font configuration
            $fontcfg  = $this->fonts[array_rand($this->fonts)];
        }

        // Full path of font file
        $fontfile = $this->resourcesPath.'/fonts/'.$fontcfg['font'];


        /** Increase font-size for shortest words: 9% for each glyp missing */
        $lettersMissing = $this->maxWordLength-strlen($text);
        $fontSizefactor = 1+($lettersMissing*0.09);

        // Text generation (char by char)
        $x      = 20*$this->scale;
        $y      = round(($this->height*27/40)*$this->scale);
        $length = strlen($text);
        for ($i=0; $i<$length; $i++) {
            $degree   = rand($this->maxRotation*-1, $this->maxRotation)*$this->difficulty;
            $fontsize = rand($fontcfg['minSize'], $fontcfg['maxSize'])*$this->scale*$fontSizefactor;
            $letter   = substr($text, $i, 1);

            if ($this->shadowColor) {
                $coords = imagettftext($this->im, $fontsize, $degree,
                    $x+$this->scale, $y+$this->scale,
                    $this->GdShadowColor, $fontfile, $letter);
            }
            $coords = imagettftext($this->im, $fontsize, $degree,
                $x, $y,
                $this->GdFgColor, $fontfile, $letter);
            $x += ($coords[2]-$x) + ($fontcfg['spacing']*$this->scale);
        }

        $this->textFinalX = $x;
    }



    /**
     * Wave filter
     */
    protected function WaveImage() {
        // create wave difficulty
        $wdf = 1;
        if($this->difficulty<1) $wdf = 1/$this->difficulty*(0.9/$this->difficulty);
        if($this->difficulty>1) $wdf = (1/($this->difficulty*1.7))+0.5;

        // X-axis wave generation
        $xp = $this->scale*$this->Xperiod*rand(1,3) * $wdf;
        $k = rand(1, 100);
        for ($i = 0; $i < ($this->width*$this->scale); $i++) {
            imagecopy($this->im, $this->im,
                $i-1, sin($k+$i/$xp) * ($this->scale*$this->Xamplitude),
                $i, 0, 1, $this->height*$this->scale);
        }

        // Y-axis wave generation
        $k = rand(0, 100);
        $yp = $this->scale*($this->Yperiod)*rand(1,2) * $wdf; 
        for ($i = 0; $i < ($this->height*$this->scale); $i++) {
            imagecopy($this->im, $this->im,
                sin($k+$i/$yp) * ($this->scale*$this->Yamplitude), $i-1,
                0, $i, $this->width*$this->scale, 1);
        }
    }




    /**
     * Reduce the image to the final size
     */
    protected function ReduceImage() {
        // Reduzco el tama�o de la imagen
        $imResampled = imagecreatetruecolor($this->width, $this->height);
        imagecopyresampled($imResampled, $this->im,
            0, 0, 0, 0,
            $this->width, $this->height,
            $this->width*$this->scale, $this->height*$this->scale
        );
        imagedestroy($this->im);
        $this->im = $imResampled;
    }








    /**
     * File generation
     */
    protected function WriteImage($text) {
        if ($this->imageFormat == 'png' && function_exists('imagepng')) {
            header("Content-type: image/png");
            imagepng($this->im, $this->dst_dir."/".$text);
        } else {
            header("Content-type: image/jpeg");
            imagejpeg($this->im, $this->dst_dir."/".$text);
        }
    }







    /**
     * Cleanup
     */
    protected function Cleanup() {
        imagedestroy($this->im);
    }
}
















?>
