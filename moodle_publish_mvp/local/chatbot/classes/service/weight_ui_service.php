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
 * Weighting UI helper service (UI-first phase).
 *
 * @package    local_chatbot
 * @copyright  2026
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */
class weight_ui_service {
    /** @var string */
    public const TABLE_SCHEME = 'local_chatbot_weight_scheme';
    /** @var string */
    public const TABLE_RULE = 'local_chatbot_weight_rule';
    /** @var string */
    public const TABLE_ACTIVITY_MAP = 'local_chatbot_weight_map';

    /** @var string */
    public const LEVEL_CATEGORY = 'category';
    /** @var string */
    public const LEVEL_TYPE = 'type';
    /** @var string */
    public const LEVEL_SOURCE = 'source';

    /** @var string */
    public const CATEGORY_TASK = 'task';
    /** @var string */
    public const CATEGORY_EXAM = 'exam';

    /** @var string */
    public const TYPE_INDIVIDUAL = 'individual';
    /** @var string */
    public const TYPE_GROUP = 'group';
    /** @var string */
    public const TYPE_PRACTICE = 'practice';
    /** @var string */
    public const TYPE_QUIZ = 'quiz';
    /** @var string */
    public const TYPE_UTS = 'uts';
    /** @var string */
    public const TYPE_UAS = 'uas';

    /** @var string */
    public const SOURCE_TEACHER = 'teacher';
    /** @var string */
    public const SOURCE_LLM = 'llm';
    /** @var string */
    public const WEIGHT_LABEL_VERY_EASY = 'very-easy';
    /** @var string */
    public const WEIGHT_LABEL_EASY = 'easy';
    /** @var string */
    public const WEIGHT_LABEL_MEDIUM = 'medium';
    /** @var string */
    public const WEIGHT_LABEL_HARD = 'hard';
    /** @var string */
    public const WEIGHT_LABEL_VERY_HARD = 'very-hard';

    /**
     * Modules supported for weighting map.
     *
     * @return array<int,string>
     */
    public static function supported_modules(): array {
        return ['assign', 'quiz'];
    }

    /**
     * Check supported module name.
     *
     * @param string $module
     * @return bool
     */
    public static function is_supported_module(string $module): bool {
        return in_array(\core_text::strtolower(trim($module)), self::supported_modules(), true);
    }

    /**
     * Get default map bucket for one activity.
     *
     * @param string $module
     * @param string $activityname
     * @return array{category:string,type:string,source:string,itemweight:float}
     */
    public static function default_map_bucket(string $module, string $activityname = ''): array {
        $normalizedmodule = \core_text::strtolower(trim($module));
        $normalizedname = \core_text::strtolower(trim($activityname));

        if ($normalizedmodule === 'quiz') {
            $type = self::TYPE_QUIZ;
            if (self::contains_any($normalizedname, ['uas', 'ujian akhir', 'final exam', 'final test'])) {
                $type = self::TYPE_UAS;
            } else if (self::contains_any($normalizedname, ['uts', 'ujian tengah', 'midterm'])) {
                $type = self::TYPE_UTS;
            } else if (self::contains_any($normalizedname, ['latihan', 'practice', 'drill', 'exercise'])) {
                $type = self::TYPE_PRACTICE;
            }

            return [
                'category' => self::CATEGORY_EXAM,
                'type' => $type,
                'source' => self::SOURCE_TEACHER,
                'itemweight' => 100.0,
            ];
        }

        $type = self::TYPE_INDIVIDUAL;
        if (self::contains_any($normalizedname, ['kelompok', 'group', 'tim', 'team'])) {
            $type = self::TYPE_GROUP;
        }

        return [
            'category' => self::CATEGORY_TASK,
            'type' => $type,
            'source' => self::SOURCE_TEACHER,
            'itemweight' => 100.0,
        ];
    }

    /**
     * Check whether all weighting tables exist.
     *
     * @return bool
     */
    public static function tables_ready(): bool {
        global $DB;

        $dbman = $DB->get_manager();
        return $dbman->table_exists(new \xmldb_table(self::TABLE_SCHEME)) &&
            $dbman->table_exists(new \xmldb_table(self::TABLE_RULE)) &&
            $dbman->table_exists(new \xmldb_table(self::TABLE_ACTIVITY_MAP));
    }

    /**
     * Category key list.
     *
     * @return array
     */
    public static function categories(): array {
        return [self::CATEGORY_TASK, self::CATEGORY_EXAM];
    }

    /**
     * Type list per category.
     *
     * @return array<string,array<int,string>>
     */
    public static function types_by_category(): array {
        return [
            self::CATEGORY_TASK => [self::TYPE_INDIVIDUAL, self::TYPE_GROUP],
            self::CATEGORY_EXAM => [self::TYPE_PRACTICE, self::TYPE_QUIZ, self::TYPE_UTS, self::TYPE_UAS],
        ];
    }

    /**
     * Source key list.
     *
     * @return array
     */
    public static function sources(): array {
        return [self::SOURCE_TEACHER, self::SOURCE_LLM];
    }

    /**
     * Weight-label to default-percent map.
     *
     * @return array<string,float>
     */
    public static function weight_label_map(): array {
        return [
            self::WEIGHT_LABEL_VERY_EASY => 30.0,
            self::WEIGHT_LABEL_EASY => 50.0,
            self::WEIGHT_LABEL_MEDIUM => 70.0,
            self::WEIGHT_LABEL_HARD => 85.0,
            self::WEIGHT_LABEL_VERY_HARD => 100.0,
        ];
    }

    /**
     * Normalize one weight label from request payload.
     *
     * @param string $label
     * @return string
     */
    public static function normalize_weight_label(string $label): string {
        $normalized = \core_text::strtolower(trim($label));
        $normalized = str_replace('_', '-', $normalized);
        if ($normalized === 'veryeasy') {
            $normalized = self::WEIGHT_LABEL_VERY_EASY;
        } else if ($normalized === 'veryhard') {
            $normalized = self::WEIGHT_LABEL_VERY_HARD;
        }
        return array_key_exists($normalized, self::weight_label_map())
            ? $normalized
            : self::WEIGHT_LABEL_MEDIUM;
    }

    /**
     * Resolve weight percent from one label.
     *
     * @param string $label
     * @return float
     */
    public static function weight_percent_from_label(string $label): float {
        $label = self::normalize_weight_label($label);
        $map = self::weight_label_map();
        return self::normalize_percent((float)$map[$label]);
    }

    /**
     * Resolve closest weight label from a percent value.
     *
     * @param float $percent
     * @return string
     */
    public static function weight_label_from_percent(float $percent): string {
        $target = self::clamp_weight_percent($percent);
        $bestlabel = self::WEIGHT_LABEL_MEDIUM;
        $bestdelta = INF;
        foreach (self::weight_label_map() as $label => $labelpercent) {
            $delta = abs($target - (float)$labelpercent);
            if ($delta < $bestdelta) {
                $bestdelta = $delta;
                $bestlabel = (string)$label;
            }
        }
        return $bestlabel;
    }

    /**
     * Clamp item weight into 0..100 range.
     *
     * @param float $value
     * @return float
     */
    public static function clamp_weight_percent(float $value): float {
        return self::normalize_percent($value);
    }

    /**
     * Default weight values.
     *
     * @return array
     */
    public static function default_weights(): array {
        $weights = [
            'category' => [
                self::CATEGORY_TASK => 40.0,
                self::CATEGORY_EXAM => 60.0,
            ],
            'type' => [
                self::CATEGORY_TASK => [
                    self::TYPE_INDIVIDUAL => 60.0,
                    self::TYPE_GROUP => 40.0,
                ],
                self::CATEGORY_EXAM => [
                    self::TYPE_PRACTICE => 15.0,
                    self::TYPE_QUIZ => 25.0,
                    self::TYPE_UTS => 25.0,
                    self::TYPE_UAS => 35.0,
                ],
            ],
            'source' => [],
        ];

        foreach (self::types_by_category() as $category => $types) {
            if (!isset($weights['source'][$category])) {
                $weights['source'][$category] = [];
            }
            foreach ($types as $type) {
                $weights['source'][$category][$type] = [
                    self::SOURCE_TEACHER => 80.0,
                    self::SOURCE_LLM => 20.0,
                ];
            }
        }

        return $weights;
    }

    /**
     * Normalize untrusted incoming weight payload.
     *
     * @param array $raw
     * @return array
     */
    public static function normalize_weights(array $raw): array {
        $weights = self::default_weights();

        foreach (self::categories() as $category) {
            $weights['category'][$category] = self::normalize_percent(
                (float)($raw['category'][$category] ?? $weights['category'][$category])
            );
            foreach (self::types_by_category()[$category] as $type) {
                $weights['type'][$category][$type] = self::normalize_percent(
                    (float)($raw['type'][$category][$type] ?? $weights['type'][$category][$type])
                );
                foreach (self::sources() as $source) {
                    $weights['source'][$category][$type][$source] = self::normalize_percent(
                        (float)($raw['source'][$category][$type][$source] ?? $weights['source'][$category][$type][$source])
                    );
                }
            }
        }

        return $weights;
    }

    /**
     * Validate weight totals.
     *
     * @param array $weights
     * @return array<int,string>
     */
    public static function validate_weights(array $weights): array {
        $errors = [];
        $epsilon = 0.0001;
        $categories = self::categories();

        $categorysum = 0.0;
        foreach ($categories as $category) {
            $categorysum += (float)($weights['category'][$category] ?? 0.0);
        }
        if (abs($categorysum - 100.0) > $epsilon) {
            $errors[] = get_string('weightserrorcategorysum', 'local_chatbot', format_float($categorysum, 2));
        }

        foreach ($categories as $category) {
            $typesum = 0.0;
            foreach (self::types_by_category()[$category] as $type) {
                $typesum += (float)($weights['type'][$category][$type] ?? 0.0);

                $sourcesum = 0.0;
                foreach (self::sources() as $source) {
                    $sourcesum += (float)($weights['source'][$category][$type][$source] ?? 0.0);
                }
                if (abs($sourcesum - 100.0) > $epsilon) {
                    $errors[] = get_string(
                        'weightserrorsourcesum',
                        'local_chatbot',
                        [
                            'bucket' => self::human_type_name($type),
                            'sum' => format_float($sourcesum, 2),
                        ]
                    );
                }
            }
            if (abs($typesum - 100.0) > $epsilon) {
                $errors[] = get_string(
                    'weightserrortypesum',
                    'local_chatbot',
                    [
                        'bucket' => self::human_category_name($category),
                        'sum' => format_float($typesum, 2),
                    ]
                );
            }
        }

        return $errors;
    }

    /**
     * Get or create active scheme for a course.
     *
     * @param int $courseid
     * @return \stdClass
     */
    public static function get_or_create_active_scheme(int $courseid): \stdClass {
        global $DB;

        $courseid = max(0, $courseid);
        if ($courseid <= 0) {
            throw new \moodle_exception('invalidcourseid', 'error');
        }

        $scheme = $DB->get_record(
            self::TABLE_SCHEME,
            ['courseid' => $courseid, 'is_active' => 1],
            '*',
            IGNORE_MISSING
        );
        if ($scheme) {
            if (!$DB->record_exists(self::TABLE_RULE, ['schemeid' => (int)$scheme->id])) {
                self::upsert_rules((int)$scheme->id, self::default_weights());
            }
            return $scheme;
        }

        $now = time();
        $record = (object)[
            'courseid' => $courseid,
            'name' => 'Default weighting scheme',
            'is_active' => 1,
            'timecreated' => $now,
            'timemodified' => $now,
        ];
        $record->id = (int)$DB->insert_record(self::TABLE_SCHEME, $record);
        self::upsert_rules((int)$record->id, self::default_weights());
        return $record;
    }

    /**
     * Load rules into weight matrix.
     *
     * @param int $schemeid
     * @return array
     */
    public static function get_scheme_weights(int $schemeid): array {
        global $DB;

        $weights = self::default_weights();
        $rules = $DB->get_records(
            self::TABLE_RULE,
            ['schemeid' => $schemeid],
            '',
            'id,level,category,subtype,source,weight_percent'
        );

        foreach ($rules as $rule) {
            $level = (string)$rule->level;
            $category = (string)$rule->category;
            $type = (string)$rule->subtype;
            $source = (string)$rule->source;
            $weight = self::normalize_percent((float)$rule->weight_percent);

            if ($level === self::LEVEL_CATEGORY && in_array($category, self::categories(), true)) {
                $weights['category'][$category] = $weight;
                continue;
            }
            if ($level === self::LEVEL_TYPE &&
                isset(self::types_by_category()[$category]) &&
                in_array($type, self::types_by_category()[$category], true)
            ) {
                $weights['type'][$category][$type] = $weight;
                continue;
            }
            if ($level === self::LEVEL_SOURCE &&
                isset(self::types_by_category()[$category]) &&
                in_array($type, self::types_by_category()[$category], true) &&
                in_array($source, self::sources(), true)
            ) {
                $weights['source'][$category][$type][$source] = $weight;
            }
        }

        return $weights;
    }

    /**
     * Persist scheme weights.
     *
     * @param int $schemeid
     * @param array $weights
     * @return void
     */
    public static function upsert_rules(int $schemeid, array $weights): void {
        global $DB;

        $schemeid = max(0, $schemeid);
        if ($schemeid <= 0) {
            return;
        }

        $now = time();
        $DB->delete_records(self::TABLE_RULE, ['schemeid' => $schemeid]);

        foreach (self::categories() as $category) {
            $DB->insert_record(self::TABLE_RULE, (object)[
                'schemeid' => $schemeid,
                'level' => self::LEVEL_CATEGORY,
                'category' => $category,
                'subtype' => 'all',
                'source' => 'all',
                'weight_percent' => self::normalize_percent((float)($weights['category'][$category] ?? 0.0)),
                'timecreated' => $now,
                'timemodified' => $now,
            ]);

            foreach (self::types_by_category()[$category] as $type) {
                $DB->insert_record(self::TABLE_RULE, (object)[
                    'schemeid' => $schemeid,
                    'level' => self::LEVEL_TYPE,
                    'category' => $category,
                    'subtype' => $type,
                    'source' => 'all',
                    'weight_percent' => self::normalize_percent((float)($weights['type'][$category][$type] ?? 0.0)),
                    'timecreated' => $now,
                    'timemodified' => $now,
                ]);

                foreach (self::sources() as $source) {
                    $DB->insert_record(self::TABLE_RULE, (object)[
                        'schemeid' => $schemeid,
                        'level' => self::LEVEL_SOURCE,
                        'category' => $category,
                        'subtype' => $type,
                        'source' => $source,
                        'weight_percent' => self::normalize_percent((float)($weights['source'][$category][$type][$source] ?? 0.0)),
                        'timecreated' => $now,
                        'timemodified' => $now,
                    ]);
                }
            }
        }

        $scheme = $DB->get_record(self::TABLE_SCHEME, ['id' => $schemeid], '*', IGNORE_MISSING);
        if ($scheme) {
            $scheme->timemodified = $now;
            $DB->update_record(self::TABLE_SCHEME, $scheme);
        }
    }

    /**
     * List assign/quiz activities from one course.
     *
     * @param int $courseid
     * @return array<int,\stdClass>
     */
    public static function get_course_activities(int $courseid): array {
        $courseid = max(0, $courseid);
        if ($courseid <= 0) {
            return [];
        }

        $modinfo = get_fast_modinfo($courseid);
        $rows = [];
        foreach ($modinfo->get_cms() as $cm) {
            if (empty($cm->id) || !in_array((string)$cm->modname, ['assign', 'quiz'], true)) {
                continue;
            }
            if (!empty($cm->deletioninprogress)) {
                continue;
            }

            $rows[] = (object)[
                'cmid' => (int)$cm->id,
                'module' => (string)$cm->modname,
                'name' => trim((string)$cm->name) !== '' ? (string)$cm->name : ('CM ' . (int)$cm->id),
                'section' => isset($cm->sectionnum) ? (int)$cm->sectionnum : 0,
                'url' => $cm->url ? $cm->url->out(false) : '',
            ];
        }

        usort($rows, static function($a, $b): int {
            $sectioncmp = (int)$a->section <=> (int)$b->section;
            if ($sectioncmp !== 0) {
                return $sectioncmp;
            }
            $modulecmp = strcmp((string)$a->module, (string)$b->module);
            if ($modulecmp !== 0) {
                return $modulecmp;
            }
            return strcmp((string)$a->name, (string)$b->name);
        });

        return $rows;
    }

    /**
     * Return saved mapping rows keyed by cmid.
     *
     * @param int $schemeid
     * @return array<int,\stdClass>
     */
    public static function get_activity_maps(int $schemeid): array {
        global $DB;

        if ($schemeid <= 0) {
            return [];
        }

        $records = $DB->get_records(self::TABLE_ACTIVITY_MAP, ['schemeid' => $schemeid], '', '*');
        $map = [];
        foreach ($records as $record) {
            $map[(int)$record->cmid] = $record;
        }
        return $map;
    }

    /**
     * Ensure one activity has default mapping in the active scheme.
     *
     * @param int $courseid
     * @param int $cmid
     * @param string $module
     * @param string $activityname
     * @return bool True if inserted new default row.
     */
    public static function ensure_default_activity_map(
        int $courseid,
        int $cmid,
        string $module,
        string $activityname = ''
    ): bool {
        global $DB;

        if ($courseid <= 0 || $cmid <= 0 || !self::tables_ready()) {
            return false;
        }

        $module = \core_text::strtolower(trim($module));
        if (!self::is_supported_module($module)) {
            return false;
        }

        $scheme = self::get_or_create_active_scheme($courseid);
        $existing = $DB->get_record(
            self::TABLE_ACTIVITY_MAP,
            ['schemeid' => (int)$scheme->id, 'cmid' => $cmid],
            '*',
            IGNORE_MISSING
        );
        if ($existing) {
            $changed = false;
            $newname = trim($activityname);
            if ((string)$existing->module !== $module) {
                $existing->module = $module;
                $changed = true;
            }
            if ($newname !== '' && (string)$existing->activityname !== $newname) {
                $existing->activityname = $newname;
                $changed = true;
            }
            if ($changed) {
                $existing->timemodified = time();
                $DB->update_record(self::TABLE_ACTIVITY_MAP, $existing);
            }
            return false;
        }

        $defaults = self::default_map_bucket($module, $activityname);
        $now = time();
        $DB->insert_record(self::TABLE_ACTIVITY_MAP, (object)[
            'schemeid' => (int)$scheme->id,
            'courseid' => $courseid,
            'cmid' => $cmid,
            'module' => $module,
            'activityname' => trim($activityname),
            'category' => $defaults['category'],
            'subtype' => $defaults['type'],
            'source' => $defaults['source'],
            'item_weight_percent' => $defaults['itemweight'],
            'timecreated' => $now,
            'timemodified' => $now,
        ]);
        return true;
    }

    /**
     * Apply mapping override from LLM-draft payload for one published activity.
     *
     * @param int $courseid
     * @param int $cmid
     * @param string $module
     * @param string $activityname
     * @param array $payload
     * @return bool
     */
    public static function apply_map_from_draft_payload(
        int $courseid,
        int $cmid,
        string $module,
        string $activityname,
        array $payload
    ): bool {
        global $DB;

        if ($courseid <= 0 || $cmid <= 0 || !self::tables_ready()) {
            return false;
        }

        $module = \core_text::strtolower(trim($module));
        if (!self::is_supported_module($module)) {
            return false;
        }

        $hasoverride = array_key_exists('weight_bucket_type', $payload) ||
            array_key_exists('weight_source', $payload) ||
            array_key_exists('activity_weight_label', $payload) ||
            array_key_exists('activity_weight_percent', $payload);
        if (!$hasoverride) {
            return self::ensure_default_activity_map($courseid, $cmid, $module, $activityname);
        }

        $defaults = self::default_map_bucket($module, $activityname);
        $category = (string)$defaults['category'];
        $type = (string)$defaults['type'];
        $source = self::SOURCE_LLM;
        $itemweight = (float)$defaults['itemweight'];

        $rawtype = \core_text::strtolower(trim((string)($payload['weight_bucket_type'] ?? '')));
        if ($rawtype !== '') {
            foreach (self::types_by_category() as $bucketcategory => $types) {
                if (in_array($rawtype, $types, true)) {
                    $category = $bucketcategory;
                    $type = $rawtype;
                    break;
                }
            }
        }

        $rawsource = \core_text::strtolower(trim((string)($payload['weight_source'] ?? '')));
        if (in_array($rawsource, self::sources(), true)) {
            $source = $rawsource;
        }

        $weightlabel = self::normalize_weight_label((string)($payload['activity_weight_label'] ?? 'medium'));
        $itemweight = self::weight_percent_from_label($weightlabel);
        if (isset($payload['activity_weight_percent']) && is_numeric((string)$payload['activity_weight_percent'])) {
            $itemweight = self::clamp_weight_percent((float)$payload['activity_weight_percent']);
        }

        if (!isset(self::types_by_category()[$category]) || !in_array($type, self::types_by_category()[$category], true)) {
            return false;
        }

        $scheme = self::get_or_create_active_scheme($courseid);
        $existing = $DB->get_record(
            self::TABLE_ACTIVITY_MAP,
            ['schemeid' => (int)$scheme->id, 'cmid' => $cmid],
            '*',
            IGNORE_MISSING
        );
        $now = time();
        if ($existing) {
            $existing->courseid = $courseid;
            $existing->module = $module;
            $existing->activityname = trim($activityname);
            $existing->category = $category;
            $existing->subtype = $type;
            $existing->source = $source;
            $existing->item_weight_percent = $itemweight;
            $existing->timemodified = $now;
            $DB->update_record(self::TABLE_ACTIVITY_MAP, $existing);
            return true;
        }

        $DB->insert_record(self::TABLE_ACTIVITY_MAP, (object)[
            'schemeid' => (int)$scheme->id,
            'courseid' => $courseid,
            'cmid' => $cmid,
            'module' => $module,
            'activityname' => trim($activityname),
            'category' => $category,
            'subtype' => $type,
            'source' => $source,
            'item_weight_percent' => $itemweight,
            'timecreated' => $now,
            'timemodified' => $now,
        ]);
        return true;
    }

    /**
     * Save activity-to-weight-bucket mapping.
     *
     * @param int $schemeid
     * @param int $courseid
     * @param array $entries
     * @param array $allowedcmids
     * @return array{saved:int,errors:array<int,string>}
     */
    public static function save_activity_maps(int $schemeid, int $courseid, array $entries, array $allowedcmids): array {
        global $DB;

        $saved = 0;
        $errors = [];
        $allowedlookup = [];
        foreach ($allowedcmids as $cmid) {
            $allowedlookup[(int)$cmid] = true;
        }

        $now = time();
        foreach ($entries as $cmid => $entry) {
            $cmid = (int)$cmid;
            if ($cmid <= 0 || !isset($allowedlookup[$cmid])) {
                continue;
            }

            $category = trim((string)($entry['category'] ?? ''));
            $type = trim((string)($entry['type'] ?? ''));
            $source = trim((string)($entry['source'] ?? ''));
            $itemweight = self::normalize_percent((float)($entry['itemweight'] ?? 100.0));
            $module = trim((string)($entry['module'] ?? ''));
            $activityname = trim((string)($entry['activityname'] ?? ''));

            if ($category === '' && $type === '' && $source === '') {
                $DB->delete_records(self::TABLE_ACTIVITY_MAP, ['schemeid' => $schemeid, 'cmid' => $cmid]);
                continue;
            }

            if (!in_array($category, self::categories(), true)) {
                $errors[] = get_string('weightserrormapinvalidcategory', 'local_chatbot', $cmid);
                continue;
            }
            if (!in_array($source, self::sources(), true)) {
                $errors[] = get_string('weightserrormapinvalidsource', 'local_chatbot', $cmid);
                continue;
            }
            if (!in_array($type, self::types_by_category()[$category], true)) {
                $errors[] = get_string('weightserrormapinvalidtype', 'local_chatbot', $cmid);
                continue;
            }

            $existing = $DB->get_record(
                self::TABLE_ACTIVITY_MAP,
                ['schemeid' => $schemeid, 'cmid' => $cmid],
                '*',
                IGNORE_MISSING
            );
            if ($existing) {
                $existing->courseid = $courseid;
                $existing->module = $module;
                $existing->activityname = $activityname;
                $existing->category = $category;
                $existing->subtype = $type;
                $existing->source = $source;
                $existing->item_weight_percent = $itemweight;
                $existing->timemodified = $now;
                $DB->update_record(self::TABLE_ACTIVITY_MAP, $existing);
                $saved++;
                continue;
            }

            $DB->insert_record(self::TABLE_ACTIVITY_MAP, (object)[
                'schemeid' => $schemeid,
                'courseid' => $courseid,
                'cmid' => $cmid,
                'module' => $module,
                'activityname' => $activityname,
                'category' => $category,
                'subtype' => $type,
                'source' => $source,
                'item_weight_percent' => $itemweight,
                'timecreated' => $now,
                'timemodified' => $now,
            ]);
            $saved++;
        }

        return ['saved' => $saved, 'errors' => $errors];
    }

    /**
     * Build preview rows for effective base weights.
     *
     * @param array $weights
     * @param array<int,\stdClass> $mapsbycmid
     * @return array<int,\stdClass>
     */
    public static function build_preview_rows(array $weights, array $mapsbycmid = []): array {
        $combo = [];
        foreach ($mapsbycmid as $map) {
            $key = (string)$map->category . '|' . (string)$map->subtype . '|' . (string)$map->source;
            if (!isset($combo[$key])) {
                $combo[$key] = 0;
            }
            $combo[$key]++;
        }

        $rows = [];
        foreach (self::categories() as $category) {
            foreach (self::types_by_category()[$category] as $type) {
                foreach (self::sources() as $source) {
                    $cat = (float)($weights['category'][$category] ?? 0.0);
                    $typ = (float)($weights['type'][$category][$type] ?? 0.0);
                    $src = (float)($weights['source'][$category][$type][$source] ?? 0.0);
                    $effective = ($cat * $typ * $src) / 10000.0;
                    $key = $category . '|' . $type . '|' . $source;
                    $rows[] = (object)[
                        'category' => $category,
                        'type' => $type,
                        'source' => $source,
                        'effective' => self::normalize_percent($effective),
                        'mappedcount' => (int)($combo[$key] ?? 0),
                    ];
                }
            }
        }
        return $rows;
    }

    /**
     * Human-friendly category label fallback.
     *
     * @param string $category
     * @return string
     */
    public static function human_category_name(string $category): string {
        $map = [
            self::CATEGORY_TASK => 'Task',
            self::CATEGORY_EXAM => 'Exam',
        ];
        return $map[$category] ?? $category;
    }

    /**
     * Human-friendly type label fallback.
     *
     * @param string $type
     * @return string
     */
    public static function human_type_name(string $type): string {
        $map = [
            self::TYPE_INDIVIDUAL => 'Individual Assignment',
            self::TYPE_GROUP => 'Group Assignment',
            self::TYPE_PRACTICE => 'Practice',
            self::TYPE_QUIZ => 'Quiz',
            self::TYPE_UTS => 'UTS',
            self::TYPE_UAS => 'UAS',
        ];
        return $map[$type] ?? $type;
    }

    /**
     * Clamp + round percent values.
     *
     * @param float $value
     * @return float
     */
    private static function normalize_percent(float $value): float {
        return (float)round(min(100.0, max(0.0, $value)), 5);
    }

    /**
     * Case-normalized contains check for any fragment.
     *
     * @param string $needlebase
     * @param array<int,string> $fragments
     * @return bool
     */
    private static function contains_any(string $needlebase, array $fragments): bool {
        if ($needlebase === '') {
            return false;
        }
        foreach ($fragments as $fragment) {
            $fragment = \core_text::strtolower(trim((string)$fragment));
            if ($fragment !== '' && strpos($needlebase, $fragment) !== false) {
                return true;
            }
        }
        return false;
    }
}
