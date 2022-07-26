<?php
$command = escapeshellcmd('python app.py');
$output = shell_exec($command);
header("Location: http://localhost/big/history/");
