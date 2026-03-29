<?php
// This file is part of Moodle - http://moodle.org/
//
// Moodle is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Moodle is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with Moodle.  If not, see <http://www.gnu.org/licenses/>.

/**
 * Publish endpoint for LLM draft -> Moodle course activity.
 *
 * @package    local_chatbot
 * @copyright  2026
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

require_once(__DIR__ . '/../../config.php');

use local_chatbot\service\draft_repository;
use local_chatbot\service\draft_validator;
use local_chatbot\service\publisher;

require_login();
require_sesskey();

$draftid = required_param('draftid', PARAM_INT);
$courseid = required_param('courseid', PARAM_INT);

$course = get_course($courseid);
require_login($course);

$context = context_course::instance($courseid);
require_capability('local/chatbot:publish', $context);
require_capability('moodle/course:manageactivities', $context);

$repository = new draft_repository();
$validator = new draft_validator();
$publisher = new publisher();

$response = ['success' => false, 'message' => ''];

try {
    $draft = $repository->get_by_id($draftid);
    if ((int)$draft->courseid !== (int)$courseid) {
        throw new moodle_exception('draftcoursemismatch', 'local_chatbot');
    }

    $payload = json_decode((string)$draft->draft_json, true);
    $contentmode = 'assignment';
    if (is_array($payload) && isset($payload['content_mode'])) {
        $normalizedmode = strtolower(trim((string)$payload['content_mode']));
        if ($normalizedmode === 'practice') {
            $contentmode = 'practice';
        }
    }

    $validator->validate_for_publish($draft);
    $published = $publisher->publish($draft, $course);
    $cmid = (int)$published['cmid'];
    $modulename = (string)$published['modulename'];
    $repository->mark_published((int)$draft->id, $cmid);

    $viewpath = '/mod/assign/view.php';
    if ($modulename === 'quiz') {
        $viewpath = '/mod/quiz/view.php';
    }

    $response = [
        'success' => true,
        'message' => ($contentmode === 'practice')
            ? get_string('practicepublishsuccess', 'local_chatbot')
            : get_string('publishsuccess', 'local_chatbot'),
        'cmid' => $cmid,
        'url' => (new moodle_url($viewpath, ['id' => $cmid]))->out(false),
    ];
} catch (Throwable $exception) {
    $repository->mark_failed($draftid, $exception->getMessage());
    $response['message'] = $exception->getMessage();
}

@header('Content-Type: application/json; charset=utf-8');
echo json_encode($response);
