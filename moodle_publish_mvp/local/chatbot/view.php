<?php
require_once(__DIR__ . '/../../config.php');
require_once(__DIR__ . '/locallib.php');

require_login();

$context = context_system::instance();
require_capability('local/chatbot:view', $context);

$filename = required_param('file', PARAM_FILE);
$datadir = local_chatbot_get_data_path();
$path = $datadir . DIRECTORY_SEPARATOR . $filename;

if (!is_file($path)) {
    http_response_code(404);
    echo 'File not found';
    exit;
}

$ext = strtolower(pathinfo($filename, PATHINFO_EXTENSION));
if ($ext !== 'pdf' && $ext !== 'txt') {
    http_response_code(400);
    echo 'Unsupported file type';
    exit;
}

if ($ext === 'pdf') {
    header('Content-Type: application/pdf');
} else {
    header('Content-Type: text/plain; charset=utf-8');
}
header('Content-Disposition: inline; filename="' . $filename . '"');
header('X-Content-Type-Options: nosniff');
header('Content-Length: ' . filesize($path));

readfile($path);
