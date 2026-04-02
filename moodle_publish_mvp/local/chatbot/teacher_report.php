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

require_once(__DIR__ . '/../../config.php');
require_once(__DIR__ . '/locallib.php');
require_once($CFG->dirroot . '/enrol/locallib.php');

require_login();

$systemcontext = context_system::instance();
require_capability('local/chatbot:view', $systemcontext);

$isteacher = local_chatbot_user_is_teacher_like((int)$USER->id) || is_siteadmin((int)$USER->id);
if (!$isteacher) {
    throw new required_capability_exception($systemcontext, 'local/chatbot:managedrafts', 'nopermissions', '');
}

/**
 * Build list of teacher-manageable courses for report filter.
 *
 * @param int $userid
 * @return array<int,\stdClass>
 */
function local_chatbot_teacher_report_courses(int $userid): array {
    global $DB;

    $courses = [];
    $enrolled = enrol_get_users_courses($userid, true, 'id,fullname,shortname', 'sortorder ASC');
    foreach ($enrolled as $course) {
        $coursecontext = context_course::instance((int)$course->id, IGNORE_MISSING);
        if (!$coursecontext) {
            continue;
        }
        if (has_capability('moodle/course:update', $coursecontext, $userid)) {
            $courses[(int)$course->id] = $course;
        }
    }

    if (is_siteadmin($userid) && local_chatbot_learning_tables_ready()) {
        $courseids = array_keys($DB->get_records_menu('local_chatbot_std_profile', null, '', 'courseid,courseid'));
        foreach ($courseids as $courseid) {
            $cid = (int)$courseid;
            if ($cid <= 0 || isset($courses[$cid])) {
                continue;
            }
            $course = $DB->get_record('course', ['id' => $cid], 'id,fullname,shortname', IGNORE_MISSING);
            if ($course) {
                $courses[$cid] = $course;
            }
        }
    }

    return $courses;
}

/**
 * Aggregate learner rows into one row per student.
 *
 * @param array $learnerrows
 * @return array
 */
function local_chatbot_teacher_report_aggregate_students(array $learnerrows): array {
    $map = [];

    foreach ($learnerrows as $row) {
        $userid = (int)($row->userid ?? 0);
        if ($userid <= 0) {
            continue;
        }

        $attempts = max(0, (int)($row->attemptsum ?? 0));
        $avgmastery = (float)($row->avgmastery ?? 0.0);
        $avgaccuracy = (float)($row->avgaccuracy ?? 0.0);
        $lastupdate = (int)($row->lastupdate ?? 0);
        $studentname = trim((string)($row->firstname ?? '') . ' ' . (string)($row->lastname ?? ''));
        if ($studentname === '') {
            $studentname = get_string('unknownuser', 'moodle');
        }

        if (!isset($map[$userid])) {
            $map[$userid] = [
                'userid' => $userid,
                'studentname' => $studentname,
                'coursecount' => 0,
                'attemptsum' => 0,
                'masteryweighted' => 0.0,
                'accuracyweighted' => 0.0,
                'masteryrowsum' => 0.0,
                'accuracyrowsum' => 0.0,
                'rowcount' => 0,
                'lastupdate' => 0,
            ];
        }

        $map[$userid]['coursecount']++;
        $map[$userid]['attemptsum'] += $attempts;
        $map[$userid]['masteryweighted'] += ($avgmastery * max(1, $attempts));
        $map[$userid]['accuracyweighted'] += ($avgaccuracy * max(1, $attempts));
        $map[$userid]['masteryrowsum'] += $avgmastery;
        $map[$userid]['accuracyrowsum'] += $avgaccuracy;
        $map[$userid]['rowcount']++;
        $map[$userid]['lastupdate'] = max((int)$map[$userid]['lastupdate'], $lastupdate);
    }

    $rows = [];
    foreach ($map as $item) {
        $weight = max(1, (int)$item['attemptsum']);
        $rowcount = max(1, (int)$item['rowcount']);
        $mastery = (int)$item['attemptsum'] > 0
            ? ((float)$item['masteryweighted'] / $weight)
            : ((float)$item['masteryrowsum'] / $rowcount);
        $accuracy = (int)$item['attemptsum'] > 0
            ? ((float)$item['accuracyweighted'] / $weight)
            : ((float)$item['accuracyrowsum'] / $rowcount);

        $rows[] = (object)[
            'userid' => (int)$item['userid'],
            'studentname' => (string)$item['studentname'],
            'coursecount' => (int)$item['coursecount'],
            'attemptsum' => (int)$item['attemptsum'],
            'mastery' => (float)$mastery,
            'accuracy' => (float)$accuracy,
            'lastupdate' => (int)$item['lastupdate'],
        ];
    }

    usort($rows, static function($a, $b): int {
        $masterycmp = (float)$a->mastery <=> (float)$b->mastery;
        if ($masterycmp !== 0) {
            return $masterycmp;
        }
        return strcmp((string)$a->studentname, (string)$b->studentname);
    });

    return $rows;
}

$courses = local_chatbot_teacher_report_courses((int)$USER->id);
$selectedcourseid = optional_param('courseid', 0, PARAM_INT);
if ($selectedcourseid > 0 && !array_key_exists($selectedcourseid, $courses)) {
    $selectedcourseid = 0;
}

$courseids = [];
if ($selectedcourseid > 0) {
    $courseids[] = $selectedcourseid;
} else {
    $courseids = array_map('intval', array_keys($courses));
}

$dataset = local_chatbot_get_teacher_mastery_dashboard($courseids);
$studentrows = local_chatbot_teacher_report_aggregate_students((array)($dataset['learners'] ?? []));

$reporturl = new moodle_url('/local/chatbot/teacher_report.php');
$reporturlwithparams = new moodle_url('/local/chatbot/teacher_report.php', ['courseid' => $selectedcourseid]);
$PAGE->set_url($reporturlwithparams);
$PAGE->set_context($systemcontext);
$PAGE->set_pagelayout('report');
$PAGE->set_title(get_string('teacherreporttitle', 'local_chatbot'));
$PAGE->set_heading(get_string('teacherreporttitle', 'local_chatbot'));

echo $OUTPUT->header();

echo $OUTPUT->heading(get_string('teacherreporttitle', 'local_chatbot'));
echo html_writer::tag('p', get_string('teacherreportsubtitle', 'local_chatbot'));

$indexurl = new moodle_url('/local/chatbot/index.php');
echo html_writer::link($indexurl, get_string('pluginname', 'local_chatbot'), ['class' => 'btn btn-secondary mb-3']);

$selectoptions = [0 => get_string('teacherreportallcourses', 'local_chatbot')];
foreach ($courses as $courseid => $course) {
    $label = trim((string)$course->fullname) !== '' ? (string)$course->fullname : (string)$course->shortname;
    $selectoptions[(int)$courseid] = $label;
}
$select = new single_select($reporturl, 'courseid', $selectoptions, $selectedcourseid);
$select->label = get_string('teacherreportfiltercourse', 'local_chatbot');
echo $OUTPUT->render($select);

$summaryitems = [
    get_string('dashboardcardstudents', 'local_chatbot') . ': ' . count($studentrows),
    get_string('dashboardcardavgmastery', 'local_chatbot') . ': ' .
        format_float((float)($dataset['summary']['avgmastery'] ?? 0.0), 1) . '%',
    get_string('dashboardcardevents', 'local_chatbot') . ': ' . (int)($dataset['summary']['eventcount'] ?? 0),
    get_string('dashboardcardlastupdate', 'local_chatbot') . ': ' .
        ((int)($dataset['summary']['lastupdate'] ?? 0) > 0
            ? userdate((int)$dataset['summary']['lastupdate'], get_string('strftimedatetime', 'langconfig'))
            : '-'),
];
echo html_writer::alist($summaryitems);

if (empty($studentrows)) {
    echo $OUTPUT->notification(get_string('teacherreportempty', 'local_chatbot'), 'info');
    echo $OUTPUT->footer();
    exit;
}

$table = new html_table();
$table->head = [
    get_string('dashboardtablestudent', 'local_chatbot'),
    get_string('dashboardtablemastery', 'local_chatbot'),
    get_string('dashboardtableaccuracy', 'local_chatbot'),
    get_string('dashboardtableattempts', 'local_chatbot'),
    get_string('teacherreportclasses', 'local_chatbot'),
    get_string('dashboardtableupdated', 'local_chatbot'),
];

$table->data = [];
foreach ($studentrows as $row) {
    $profileurl = new moodle_url('/user/profile.php', ['id' => (int)$row->userid]);
    $table->data[] = [
        html_writer::link($profileurl, s((string)$row->studentname)),
        s(format_float((float)$row->mastery, 1) . '%'),
        s(format_float((float)$row->accuracy, 1) . '%'),
        (int)$row->attemptsum,
        (int)$row->coursecount,
        s((int)$row->lastupdate > 0
            ? userdate((int)$row->lastupdate, get_string('strftimedatetime', 'langconfig'))
            : '-'),
    ];
}

echo html_writer::table($table);
echo $OUTPUT->footer();
