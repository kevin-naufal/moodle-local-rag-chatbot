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

defined('MOODLE_INTERNAL') || die();

$observers = [
    [
        'eventname' => '\core\event\course_module_created',
        'callback' => '\local_chatbot\observer::course_module_created',
        'priority' => 9999,
    ],
    [
        'eventname' => '\mod_quiz\event\attempt_submitted',
        'callback' => '\local_chatbot\observer::quiz_attempt_submitted',
        'priority' => 9999,
    ],
    [
        'eventname' => '\mod_assign\event\assessable_submitted',
        'callback' => '\local_chatbot\observer::assign_assessable_submitted',
        'priority' => 9999,
    ],
    [
        'eventname' => '\mod_assign\event\submission_graded',
        'callback' => '\local_chatbot\observer::assign_submission_graded',
        'priority' => 9999,
    ],
];
