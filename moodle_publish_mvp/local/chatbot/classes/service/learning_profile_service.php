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

namespace local_chatbot\service;

defined('MOODLE_INTERNAL') || die();

/**
 * Capture learning events and maintain student profile by topic.
 *
 * @package    local_chatbot
 * @copyright  2026
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */
class learning_profile_service {
    /** @var string */
    private const PROFILE_TABLE = 'local_chatbot_std_profile';
    /** @var string */
    private const EVENTS_TABLE = 'local_chatbot_learn_events';
    /** @var string */
    private const MODULE_QUIZ = 'quiz';

    /**
     * Ingest one submitted quiz attempt into learning events + profile.
     *
     * @param int $attemptid
     * @param int $cmid
     * @param int $courseid
     * @return void
     */
    public static function ingest_quiz_attempt(int $attemptid, int $cmid, int $courseid): void {
        global $DB;

        if ($attemptid <= 0 || $courseid <= 0) {
            return;
        }

        $attempt = $DB->get_record(
            'quiz_attempts',
            ['id' => $attemptid],
            'id,quiz,userid,attempt,timestart,timefinish,sumgrades,state',
            IGNORE_MISSING
        );
        if (!$attempt || (int)$attempt->userid <= 0) {
            return;
        }

        $quiz = $DB->get_record(
            'quiz',
            ['id' => (int)$attempt->quiz],
            'id,course,name,intro,attempts,sumgrades',
            IGNORE_MISSING
        );
        if (!$quiz) {
            return;
        }

        $resolvedcourseid = (int)$quiz->course > 0 ? (int)$quiz->course : $courseid;
        $topic = self::resolve_topic($cmid);
        $eventtype = self::resolve_event_type($quiz);

        $scoremax = max((float)$quiz->sumgrades, 0.0);
        $scoreraw = max((float)$attempt->sumgrades, 0.0);
        $scoretopic = self::clamp_percent($scoremax > 0 ? ($scoreraw / $scoremax) * 100.0 : 0.0);

        $startedat = (int)$attempt->timestart;
        $submittedat = (int)$attempt->timefinish;
        if ($submittedat <= 0) {
            $submittedat = time();
        }

        $duration = 0;
        if ($startedat > 0 && (int)$attempt->timefinish >= $startedat) {
            $duration = (int)$attempt->timefinish - $startedat;
        }

        $details = [
            'state' => (string)$attempt->state,
            'quizname' => (string)$quiz->name,
            'ispractice' => $eventtype === 'practice',
        ];

        try {
            $existingevent = $DB->get_record(
                self::EVENTS_TABLE,
                ['module' => self::MODULE_QUIZ, 'attemptid' => $attemptid],
                '*',
                IGNORE_MISSING
            );

            if ($existingevent) {
                $oldtopic = (string)$existingevent->topic;
                $updaterecord = (object)[
                    'id' => (int)$existingevent->id,
                    'userid' => (int)$attempt->userid,
                    'courseid' => $resolvedcourseid,
                    'topic' => $topic,
                    'module' => self::MODULE_QUIZ,
                    'event_type' => $eventtype,
                    'cmid' => $cmid > 0 ? $cmid : null,
                    'activityid' => (int)$quiz->id,
                    'attemptid' => (int)$attempt->id,
                    'attempt_number' => (int)$attempt->attempt,
                    'score_raw' => self::round_num($scoreraw, 5),
                    'score_max' => self::round_num($scoremax, 5),
                    'score_topic' => self::round_num($scoretopic, 5),
                    'duration_seconds' => $duration,
                    'started_at' => $startedat > 0 ? $startedat : null,
                    'submitted_at' => $submittedat,
                    'details_json' => json_encode($details),
                ];
                $DB->update_record(self::EVENTS_TABLE, $updaterecord);

                self::recompute_profile_from_events((int)$attempt->userid, $resolvedcourseid, $topic);
                if ($oldtopic !== '' && $oldtopic !== $topic) {
                    self::recompute_profile_from_events((int)$attempt->userid, $resolvedcourseid, $oldtopic);
                }
                return;
            }

            $eventrecord = (object)[
                'userid' => (int)$attempt->userid,
                'courseid' => $resolvedcourseid,
                'topic' => $topic,
                'module' => self::MODULE_QUIZ,
                'event_type' => $eventtype,
                'cmid' => $cmid > 0 ? $cmid : null,
                'activityid' => (int)$quiz->id,
                'attemptid' => (int)$attempt->id,
                'attempt_number' => (int)$attempt->attempt,
                'score_raw' => self::round_num($scoreraw, 5),
                'score_max' => self::round_num($scoremax, 5),
                'score_topic' => self::round_num($scoretopic, 5),
                'duration_seconds' => $duration,
                'started_at' => $startedat > 0 ? $startedat : null,
                'submitted_at' => $submittedat,
                'details_json' => json_encode($details),
                'timecreated' => time(),
            ];
            $DB->insert_record(self::EVENTS_TABLE, $eventrecord);

            self::upsert_profile(
                (int)$attempt->userid,
                $resolvedcourseid,
                $topic,
                $scoretopic,
                $duration,
                $submittedat
            );
        } catch (\dml_write_exception $e) {
            // Duplicate attempt log can happen in rare race conditions; ignore safely.
            return;
        }
    }

    /**
     * Recompute one profile aggregate from existing events.
     *
     * @param int $userid
     * @param int $courseid
     * @param string $topic
     * @return void
     */
    private static function recompute_profile_from_events(int $userid, int $courseid, string $topic): void {
        global $DB;

        if ($userid <= 0 || $courseid <= 0 || trim($topic) === '') {
            return;
        }

        $events = $DB->get_records(
            self::EVENTS_TABLE,
            ['userid' => $userid, 'courseid' => $courseid, 'topic' => $topic],
            'submitted_at ASC, id ASC',
            'id,score_topic,duration_seconds,submitted_at'
        );

        if (!$events) {
            $DB->delete_records(self::PROFILE_TABLE, ['userid' => $userid, 'courseid' => $courseid, 'topic' => $topic]);
            return;
        }

        $scores = [];
        $durationsum = 0.0;
        $lastscore = 0.0;
        $lasttime = 0;
        foreach ($events as $event) {
            $score = self::clamp_percent((float)$event->score_topic);
            $scores[] = $score;
            $durationsum += max((float)$event->duration_seconds, 0.0);
            $lastscore = $score;
            $lasttime = max($lasttime, (int)$event->submitted_at);
        }

        $attemptcount = count($scores);
        if ($attemptcount <= 0) {
            return;
        }

        $accuracyavg = array_sum($scores) / $attemptcount;
        $avgduration = $durationsum / $attemptcount;

        $mastery = (float)$scores[0];
        $trend = 0.0;
        $previous = (float)$scores[0];
        for ($i = 1; $i < $attemptcount; $i++) {
            $score = (float)$scores[$i];
            $mastery = (0.7 * $mastery) + (0.3 * $score);
            $delta = $score - $previous;
            $trend = (0.7 * $trend) + (0.3 * $delta);
            $previous = $score;
        }

        $now = time();
        $profile = $DB->get_record(
            self::PROFILE_TABLE,
            ['userid' => $userid, 'courseid' => $courseid, 'topic' => $topic],
            '*',
            IGNORE_MISSING
        );

        if (!$profile) {
            $profile = (object)[
                'userid' => $userid,
                'courseid' => $courseid,
                'topic' => $topic,
                'timecreated' => $now,
            ];
        }

        $profile->mastery = self::round_num(self::clamp_percent($mastery), 5);
        $profile->accuracy_avg = self::round_num(self::clamp_percent($accuracyavg), 5);
        $profile->attempt_count = $attemptcount;
        $profile->avg_duration_seconds = self::round_num(max($avgduration, 0.0), 2);
        $profile->trend = self::round_num($trend, 5);
        $profile->last_score = self::round_num(self::clamp_percent($lastscore), 5);
        $profile->last_event_time = $lasttime > 0 ? $lasttime : $now;
        $profile->timemodified = $now;

        if (!empty($profile->id)) {
            $DB->update_record(self::PROFILE_TABLE, $profile);
            return;
        }

        $DB->insert_record(self::PROFILE_TABLE, $profile);
    }

    /**
     * Upsert one student profile aggregate entry.
     *
     * @param int $userid
     * @param int $courseid
     * @param string $topic
     * @param float $scoretopic
     * @param int $duration
     * @param int $submittedat
     * @return void
     */
    private static function upsert_profile(
        int $userid,
        int $courseid,
        string $topic,
        float $scoretopic,
        int $duration,
        int $submittedat
    ): void {
        global $DB;

        $now = time();
        $profile = $DB->get_record(
            self::PROFILE_TABLE,
            ['userid' => $userid, 'courseid' => $courseid, 'topic' => $topic],
            '*',
            IGNORE_MISSING
        );

        if (!$profile) {
            $record = (object)[
                'userid' => $userid,
                'courseid' => $courseid,
                'topic' => $topic,
                'mastery' => self::round_num($scoretopic, 5),
                'accuracy_avg' => self::round_num($scoretopic, 5),
                'attempt_count' => 1,
                'avg_duration_seconds' => self::round_num((float)$duration, 2),
                'trend' => 0.0,
                'last_score' => self::round_num($scoretopic, 5),
                'last_event_time' => $submittedat,
                'timecreated' => $now,
                'timemodified' => $now,
            ];
            $DB->insert_record(self::PROFILE_TABLE, $record);
            return;
        }

        $oldattempts = max((int)$profile->attempt_count, 0);
        $newattempts = $oldattempts + 1;
        $oldaccuracy = (float)$profile->accuracy_avg;
        $oldavgduration = (float)$profile->avg_duration_seconds;
        $oldmastery = (float)$profile->mastery;
        $oldtrend = (float)$profile->trend;
        $oldlastscore = (float)$profile->last_score;

        $accuracyavg = (($oldaccuracy * $oldattempts) + $scoretopic) / max($newattempts, 1);
        $durationavg = (($oldavgduration * $oldattempts) + max($duration, 0)) / max($newattempts, 1);

        // Sprint 1 formula: mastery_new = 0.7*mastery_old + 0.3*score_topic.
        $masterynew = (0.7 * $oldmastery) + (0.3 * $scoretopic);
        $delta = $scoretopic - $oldlastscore;
        $trendnew = (0.7 * $oldtrend) + (0.3 * $delta);

        $profile->mastery = self::round_num(self::clamp_percent($masterynew), 5);
        $profile->accuracy_avg = self::round_num(self::clamp_percent($accuracyavg), 5);
        $profile->attempt_count = $newattempts;
        $profile->avg_duration_seconds = self::round_num(max($durationavg, 0.0), 2);
        $profile->trend = self::round_num($trendnew, 5);
        $profile->last_score = self::round_num(self::clamp_percent($scoretopic), 5);
        $profile->last_event_time = $submittedat;
        $profile->timemodified = $now;

        $DB->update_record(self::PROFILE_TABLE, $profile);
    }

    /**
     * Resolve topic label from course section.
     *
     * @param int $cmid
     * @return string
     */
    private static function resolve_topic(int $cmid): string {
        global $DB;

        if ($cmid <= 0) {
            return 'General';
        }

        $cm = $DB->get_record('course_modules', ['id' => $cmid], 'id,section', IGNORE_MISSING);
        if (!$cm || empty($cm->section)) {
            return 'General';
        }

        $section = $DB->get_record(
            'course_sections',
            ['id' => (int)$cm->section],
            'id,section,name',
            IGNORE_MISSING
        );
        if (!$section) {
            return 'General';
        }

        $name = trim((string)$section->name);
        if ($name !== '') {
            return self::normalize_topic($name);
        }

        if ((int)$section->section > 0) {
            return 'Topic ' . (int)$section->section;
        }

        return 'General';
    }

    /**
     * Normalize topic text to keep profile keys stable.
     *
     * @param string $topic
     * @return string
     */
    private static function normalize_topic(string $topic): string {
        $text = html_entity_decode($topic, ENT_QUOTES | ENT_HTML5, 'UTF-8');
        $text = strip_tags($text);
        $text = preg_replace('/\s+/', ' ', trim($text));
        if (!is_string($text) || $text === '') {
            return 'General';
        }
        return $text;
    }

    /**
     * Determine whether attempt should be tagged as quiz or practice.
     *
     * @param \stdClass $quiz
     * @return string
     */
    private static function resolve_event_type(\stdClass $quiz): string {
        if ((int)$quiz->attempts === 0) {
            return 'practice';
        }

        if (stripos((string)$quiz->name, '[Practice]') === 0) {
            return 'practice';
        }

        $intro = trim(strip_tags((string)$quiz->intro));
        if ($intro !== '' && stripos($intro, 'Practice Quiz') !== false) {
            return 'practice';
        }

        return 'quiz';
    }

    /**
     * Clamp percent values into 0..100.
     *
     * @param float $value
     * @return float
     */
    private static function clamp_percent(float $value): float {
        return min(100.0, max(0.0, $value));
    }

    /**
     * Round helper.
     *
     * @param float $value
     * @param int $decimals
     * @return float
     */
    private static function round_num(float $value, int $decimals): float {
        return (float)round($value, $decimals);
    }
}
