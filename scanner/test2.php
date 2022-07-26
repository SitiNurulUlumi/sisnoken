<?php
$command = escapeshellcmd('python app2.py');
$output = shell_exec($command);
header("Location: http://localhost/big/historykeluar/");
