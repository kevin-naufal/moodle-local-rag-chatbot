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
 * Essay auto-grading service (v1 rubric contract).
 *
 * @package    local_chatbot
 * @copyright  2026
 * @license    http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */
class essay_autograder {
    /** @var string */
    private const VERSION = 'essay_autograde_output_v1';

    /** @var string */
    private const DEFAULT_RUBRIC_ID = 'essay_default_v1';

    /** @var float */
    private const MAX_SCORE = 100.0;

    /** @var array<string,float> */
    private const CRITERIA = [
        'content_accuracy' => 35.0,
        'coverage_of_key_points' => 30.0,
        'reasoning_quality' => 20.0,
        'organization_clarity' => 10.0,
        'language_mechanics' => 5.0,
    ];

    /**
     * Run essay auto-grading and return normalized JSON-ready output.
     *
     * @param array $input
     * @return array
     */
    public function grade(array $input): array {
        global $CFG;

        if (!function_exists('local_chatbot_run_llm_general')) {
            require_once($CFG->dirroot . '/local/chatbot/locallib.php');
        }

        $normalizedinput = $this->normalize_input($input);
        $prompt = $this->build_grading_prompt($normalizedinput);
        $modelanswer = '';
        try {
            $result = \local_chatbot_run_llm_general($prompt, true);
            $modelanswer = (string)($result['answer'] ?? '');
            $parsed = $this->extract_model_json($modelanswer);
        } catch (\Throwable $e) {
            $parsed = $this->build_fallback_raw_output($normalizedinput, $e->getMessage(), $modelanswer);
        }
        return $this->normalize_output($parsed, $normalizedinput);
    }

    /**
     * Validate and normalize grading input.
     *
     * @param array $input
     * @return array
     */
    private function normalize_input(array $input): array {
        $questiontext = trim((string)($input['question_text'] ?? ''));
        $expectedkeypoints = trim((string)($input['expected_key_points'] ?? ''));
        $studentanswer = trim((string)($input['student_answer'] ?? ''));
        $questionnumber = max(1, (int)($input['question_number'] ?? 1));
        $rubricid = trim((string)($input['rubric_id'] ?? self::DEFAULT_RUBRIC_ID));

        if ($questiontext === '') {
            throw new \invalid_parameter_exception('question_text is required.');
        }
        if ($expectedkeypoints === '') {
            throw new \invalid_parameter_exception('expected_key_points is required.');
        }
        if ($rubricid === '') {
            $rubricid = self::DEFAULT_RUBRIC_ID;
        }

        return [
            'question_text' => $questiontext,
            'expected_key_points' => $expectedkeypoints,
            'student_answer' => $studentanswer,
            'question_number' => $questionnumber,
            'rubric_id' => $rubricid,
        ];
    }

    /**
     * Build strict grading prompt with JSON-only contract.
     *
     * @param array $input
     * @return string
     */
    private function build_grading_prompt(array $input): string {
        $criteria = [];
        foreach (self::CRITERIA as $key => $weight) {
            $criteria[] = '- ' . $key . ': weight=' . (int)$weight . ', level=0..4';
        }

        $promptparts = [
            'You are an academic essay grader.',
            'Evaluate the student answer based ONLY on the question and expected key points below.',
            'Return JSON only. No markdown. No explanation outside JSON.',
            'Use rubric_id exactly as provided.',
            'Use criteria and weights:',
            implode("\n", $criteria),
            'Scoring rule: score = (level / 4) * weight.',
            'Overall score must be sum of criterion scores.',
            'If answer is too short, irrelevant, unsafe, or uncertain, set flags.needs_manual_review=true and explain in flags.reasons.',
            'Required JSON shape:',
            '{',
            '  "version": "essay_autograde_output_v1",',
            '  "rubric_id": "<string>",',
            '  "question_number": <int>,',
            '  "overall_score": <0..100 number>,',
            '  "max_score": 100,',
            '  "criterion_scores": [',
            '    {"criterion_key":"content_accuracy","weight":35,"level":0,"score":0,"reason":"..."},',
            '    {"criterion_key":"coverage_of_key_points","weight":30,"level":0,"score":0,"reason":"..."},',
            '    {"criterion_key":"reasoning_quality","weight":20,"level":0,"score":0,"reason":"..."},',
            '    {"criterion_key":"organization_clarity","weight":10,"level":0,"score":0,"reason":"..."},',
            '    {"criterion_key":"language_mechanics","weight":5,"level":0,"score":0,"reason":"..."}',
            '  ],',
            '  "strengths": ["..."],',
            '  "improvement_suggestions": ["..."],',
            '  "missing_key_points": ["..."],',
            '  "flags": {"needs_manual_review": false, "reasons": []},',
            '  "confidence": <0..1 number>,',
            '  "grader_notes": "..."',
            '}',
            'Question number: ' . (int)$input['question_number'],
            'Rubric ID: ' . $input['rubric_id'],
            'Question text:',
            $input['question_text'],
            'Expected key points:',
            $input['expected_key_points'],
            'Student answer:',
            $input['student_answer'] !== '' ? $input['student_answer'] : '[EMPTY_ANSWER]',
        ];

        return implode("\n\n", $promptparts);
    }

    /**
     * Parse model answer and extract first valid JSON object.
     *
     * @param string $answer
     * @return array
     */
    private function extract_model_json(string $answer): array {
        $trimmed = trim($answer);
        if ($trimmed === '') {
            throw new \Exception('Auto-grader returned empty response.');
        }

        $decoded = json_decode($trimmed, true);
        if (is_array($decoded)) {
            return $decoded;
        }

        if (preg_match('/```(?:json)?\s*(\{[\s\S]*\})\s*```/i', $trimmed, $matches)) {
            $decoded = json_decode((string)$matches[1], true);
            if (is_array($decoded)) {
                return $decoded;
            }
        }

        $jsonfragment = $this->find_first_balanced_json_object($trimmed);
        if ($jsonfragment !== '') {
            $decoded = json_decode($jsonfragment, true);
            if (is_array($decoded)) {
                return $decoded;
            }
        }

        throw new \Exception('Auto-grader did not return valid JSON payload.');
    }

    /**
     * Find first balanced JSON object in arbitrary text.
     *
     * @param string $text
     * @return string
     */
    private function find_first_balanced_json_object(string $text): string {
        $start = -1;
        $depth = 0;
        $instring = false;
        $escape = false;
        $length = \core_text::strlen($text);

        for ($i = 0; $i < $length; $i++) {
            $ch = \core_text::substr($text, $i, 1);
            if ($escape) {
                $escape = false;
                continue;
            }
            if ($ch === '\\') {
                $escape = true;
                continue;
            }
            if ($ch === '"') {
                $instring = !$instring;
                continue;
            }
            if ($instring) {
                continue;
            }

            if ($ch === '{') {
                if ($depth === 0) {
                    $start = $i;
                }
                $depth++;
                continue;
            }
            if ($ch === '}' && $depth > 0) {
                $depth--;
                if ($depth === 0 && $start >= 0) {
                    return (string)\core_text::substr($text, $start, ($i - $start + 1));
                }
            }
        }

        return '';
    }

    /**
     * Normalize model output to v1 schema and enforce deterministic scoring.
     *
     * @param array $raw
     * @param array $input
     * @return array
     */
    private function normalize_output(array $raw, array $input): array {
        $criterionmap = [];
        $rawcriteria = $raw['criterion_scores'] ?? [];
        if (is_array($rawcriteria)) {
            foreach ($rawcriteria as $item) {
                if (!is_array($item)) {
                    continue;
                }
                $key = trim((string)($item['criterion_key'] ?? ''));
                if ($key !== '') {
                    $criterionmap[$key] = $item;
                }
            }
        }

        $criterionscores = [];
        $totalscore = 0.0;
        foreach (self::CRITERIA as $criterionkey => $weight) {
            $rawitem = $criterionmap[$criterionkey] ?? [];
            $level = $this->clamp_int((int)($rawitem['level'] ?? 0), 0, 4);
            $score = round(($level / 4.0) * $weight, 2);
            $reason = $this->sanitize_text((string)($rawitem['reason'] ?? ''), 500);
            if ($reason === '') {
                $reason = 'No detailed justification provided.';
            }

            $criterionscores[] = [
                'criterion_key' => $criterionkey,
                'weight' => $weight,
                'level' => $level,
                'score' => $score,
                'reason' => $reason,
            ];
            $totalscore += $score;
        }

        $forcedmanualreasons = [];
        $wordcount = $this->count_words((string)($input['student_answer'] ?? ''));
        if ($wordcount < 20) {
            // Essay answers that are too short are not gradable reliably.
            foreach ($criterionscores as &$item) {
                $item['level'] = 0;
                $item['score'] = 0.0;
                $item['reason'] = 'Score forced to 0 because answer is too short for essay grading.';
            }
            unset($item);
            $totalscore = 0.0;
            $forcedmanualreasons[] = 'Answer too short (<20 words), auto-score forced to 0.';
        }

        $strengths = $this->sanitize_list($raw['strengths'] ?? [], 5, 300);
        $improvements = $this->sanitize_list($raw['improvement_suggestions'] ?? [], 5, 300);
        $missing = $this->sanitize_list($raw['missing_key_points'] ?? [], 10, 300);
        $flagreasons = $this->sanitize_list(
            isset($raw['flags']['reasons']) ? $raw['flags']['reasons'] : [],
            5,
            200
        );

        $confidence = $this->clamp_float((float)($raw['confidence'] ?? 0.7), 0.0, 1.0);
        $manualreasons = $this->derive_manual_review_reasons($input, $confidence);
        foreach ($manualreasons as $reason) {
            if (!in_array($reason, $flagreasons, true)) {
                $flagreasons[] = $reason;
            }
        }
        foreach ($forcedmanualreasons as $reason) {
            if (!in_array($reason, $flagreasons, true)) {
                $flagreasons[] = $reason;
            }
        }
        if (!empty($forcedmanualreasons)) {
            $confidence = min($confidence, 0.3);
        }

        $needsmanualreview = !empty($flagreasons) || !empty($raw['flags']['needs_manual_review']);
        $gradersnotes = $this->sanitize_text((string)($raw['grader_notes'] ?? ''), 1000);

        return [
            'version' => self::VERSION,
            'rubric_id' => $input['rubric_id'],
            'question_number' => (int)$input['question_number'],
            'overall_score' => round($totalscore, 2),
            'max_score' => self::MAX_SCORE,
            'criterion_scores' => $criterionscores,
            'strengths' => $strengths,
            'improvement_suggestions' => $improvements,
            'missing_key_points' => $missing,
            'flags' => [
                'needs_manual_review' => $needsmanualreview,
                'reasons' => array_slice($flagreasons, 0, 5),
            ],
            'confidence' => round($confidence, 4),
            'grader_notes' => $gradersnotes,
        ];
    }

    /**
     * Derive deterministic review flags independent from model wording.
     *
     * @param array $input
     * @param float $confidence
     * @return array
     */
    private function derive_manual_review_reasons(array $input, float $confidence): array {
        $reasons = [];
        $answer = trim((string)$input['student_answer']);
        if ($answer === '' || $answer === '[EMPTY_ANSWER]') {
            $reasons[] = 'Student answer is empty.';
            return $reasons;
        }

        $wordcount = $this->count_words($answer);
        if ($wordcount < 40) {
            $reasons[] = 'Student answer is too short for reliable auto-grading.';
        }
        if ($confidence < 0.55) {
            $reasons[] = 'Model confidence is below threshold (0.55).';
        }
        return $reasons;
    }

    /**
     * Build fallback raw output when LLM grading response is unavailable/invalid.
     *
     * @param array $input
     * @param string $error
     * @param string $modelanswer
     * @return array
     */
    private function build_fallback_raw_output(array $input, string $error, string $modelanswer = ''): array {
        $answer = trim((string)($input['student_answer'] ?? ''));
        $wordcount = $this->count_words($answer);
        $points = $this->extract_expected_points((string)($input['expected_key_points'] ?? ''));

        $covered = 0;
        $missing = [];
        foreach ($points as $point) {
            if ($this->is_point_covered($point, $answer)) {
                $covered++;
            } else {
                $missing[] = $point;
            }
        }
        $totalpoints = count($points);
        $coverageratio = $totalpoints > 0 ? ($covered / $totalpoints) : 0.0;

        $contentlevel = $this->clamp_int((int)round($coverageratio * 4), 0, 4);
        $coveragelevel = $this->clamp_int((int)floor($coverageratio * 4), 0, 4);
        $reasoninglevel = $this->derive_level_from_word_count($wordcount);
        $organizationlevel = $this->derive_organization_level($answer, $wordcount);
        $languagelevel = $this->derive_language_level($answer, $wordcount);

        $notes = 'Fallback scorer used because model output was invalid or unavailable.';
        $error = trim($error);
        if ($error !== '') {
            $notes .= ' Error: ' . $this->sanitize_text($error, 220);
        }
        if (trim($modelanswer) !== '') {
            $notes .= ' Model answer snapshot: ' . $this->sanitize_text($modelanswer, 200);
        }

        $strengths = [];
        if ($covered > 0) {
            $strengths[] = 'Answer addresses ' . $covered . ' key point(s).';
        }
        if ($wordcount >= 80) {
            $strengths[] = 'Answer length is sufficient for basic argument development.';
        }
        if (empty($strengths)) {
            $strengths[] = 'Submission is present.';
        }

        $improvements = [];
        if (!empty($missing)) {
            $improvements[] = 'Cover missing key points from the expected answer.';
        }
        if ($wordcount < 80) {
            $improvements[] = 'Add more explanation and examples to strengthen reasoning.';
        }
        $improvements[] = 'Use clearer structure (opening, analysis, conclusion).';

        return [
            'version' => self::VERSION,
            'rubric_id' => (string)$input['rubric_id'],
            'question_number' => (int)$input['question_number'],
            'overall_score' => 0,
            'max_score' => self::MAX_SCORE,
            'criterion_scores' => [
                [
                    'criterion_key' => 'content_accuracy',
                    'weight' => self::CRITERIA['content_accuracy'],
                    'level' => $contentlevel,
                    'score' => 0,
                    'reason' => 'Estimated from key-point coverage ratio: ' . round($coverageratio * 100, 1) . '%.',
                ],
                [
                    'criterion_key' => 'coverage_of_key_points',
                    'weight' => self::CRITERIA['coverage_of_key_points'],
                    'level' => $coveragelevel,
                    'score' => 0,
                    'reason' => 'Covered ' . $covered . ' of ' . max($totalpoints, 1) . ' expected key points.',
                ],
                [
                    'criterion_key' => 'reasoning_quality',
                    'weight' => self::CRITERIA['reasoning_quality'],
                    'level' => $reasoninglevel,
                    'score' => 0,
                    'reason' => 'Estimated using answer length and explanation depth.',
                ],
                [
                    'criterion_key' => 'organization_clarity',
                    'weight' => self::CRITERIA['organization_clarity'],
                    'level' => $organizationlevel,
                    'score' => 0,
                    'reason' => 'Estimated from paragraph and sentence structure.',
                ],
                [
                    'criterion_key' => 'language_mechanics',
                    'weight' => self::CRITERIA['language_mechanics'],
                    'level' => $languagelevel,
                    'score' => 0,
                    'reason' => 'Estimated from basic text quality indicators.',
                ],
            ],
            'strengths' => $strengths,
            'improvement_suggestions' => $improvements,
            'missing_key_points' => array_slice($missing, 0, 10),
            'flags' => [
                'needs_manual_review' => true,
                'reasons' => [
                    'Fallback scoring used due invalid/unavailable model JSON output.',
                ],
            ],
            'confidence' => 0.45,
            'grader_notes' => $notes,
        ];
    }

    /**
     * Split expected key points text into concise bullet-like points.
     *
     * @param string $text
     * @return array
     */
    private function extract_expected_points(string $text): array {
        $normalized = str_replace(["\r\n", "\r"], "\n", trim($text));
        if ($normalized === '') {
            return [];
        }

        $chunks = preg_split('/\n+|;\s+|(?<=\.)\s+(?=[A-Z0-9\-])/u', $normalized);
        if (!is_array($chunks)) {
            return [];
        }

        $points = [];
        foreach ($chunks as $chunk) {
            $item = trim((string)$chunk);
            $item = preg_replace('/^[-*0-9\.\)\s]+/u', '', $item);
            if (!is_string($item)) {
                continue;
            }
            $item = trim($item);
            if ($item !== '') {
                $points[] = $item;
            }
        }
        return array_slice(array_values(array_unique($points)), 0, 20);
    }

    /**
     * Heuristic coverage check for one expected point against student answer.
     *
     * @param string $point
     * @param string $answer
     * @return bool
     */
    private function is_point_covered(string $point, string $answer): bool {
        $point = trim($point);
        $answer = trim($answer);
        if ($point === '' || $answer === '') {
            return false;
        }

        $answerlower = \core_text::strtolower($answer);
        $pointlower = \core_text::strtolower($point);
        if (\core_text::strlen($pointlower) >= 12 && strpos($answerlower, $pointlower) !== false) {
            return true;
        }

        preg_match_all('/\p{L}[\p{L}\p{N}\-_]*/u', $pointlower, $matches);
        $tokens = array_values(array_filter($matches[0] ?? [], function(string $token): bool {
            return \core_text::strlen($token) >= 4;
        }));
        if (empty($tokens)) {
            return false;
        }

        $found = 0;
        foreach ($tokens as $token) {
            if (strpos($answerlower, $token) !== false) {
                $found++;
            }
        }

        $ratio = $found / max(count($tokens), 1);
        return $ratio >= 0.4 || $found >= 2;
    }

    /**
     * Derive level (0..4) from word count.
     *
     * @param int $wordcount
     * @return int
     */
    private function derive_level_from_word_count(int $wordcount): int {
        if ($wordcount <= 0) {
            return 0;
        }
        if ($wordcount < 40) {
            return 1;
        }
        if ($wordcount < 80) {
            return 2;
        }
        if ($wordcount < 140) {
            return 3;
        }
        return 4;
    }

    /**
     * Derive organization clarity level from structure signals.
     *
     * @param string $answer
     * @param int $wordcount
     * @return int
     */
    private function derive_organization_level(string $answer, int $wordcount): int {
        if ($wordcount <= 0) {
            return 0;
        }
        $paragraphs = preg_split('/\n{2,}/', trim($answer));
        $paragraphcount = is_array($paragraphs) ? count(array_filter($paragraphs, 'strlen')) : 1;
        if ($paragraphcount >= 3 && $wordcount >= 100) {
            return 4;
        }
        if ($paragraphcount >= 2 && $wordcount >= 70) {
            return 3;
        }
        if ($wordcount >= 40) {
            return 2;
        }
        return 1;
    }

    /**
     * Derive language mechanics level using basic textual quality checks.
     *
     * @param string $answer
     * @param int $wordcount
     * @return int
     */
    private function derive_language_level(string $answer, int $wordcount): int {
        if ($wordcount <= 0) {
            return 0;
        }
        $sentences = preg_split('/[.!?]+/u', trim($answer));
        $sentencecount = is_array($sentences) ? count(array_filter($sentences, 'strlen')) : 1;
        if ($sentencecount >= 4 && $wordcount >= 90) {
            return 4;
        }
        if ($sentencecount >= 3 && $wordcount >= 60) {
            return 3;
        }
        if ($sentencecount >= 2 && $wordcount >= 30) {
            return 2;
        }
        return 1;
    }

    /**
     * Count words in UTF-8 text.
     *
     * @param string $text
     * @return int
     */
    private function count_words(string $text): int {
        preg_match_all('/\p{L}[\p{L}\p{N}\-_]*/u', $text, $matches);
        return count($matches[0] ?? []);
    }

    /**
     * Sanitize list of strings with max items and max length.
     *
     * @param mixed $value
     * @param int $maxitems
     * @param int $maxlength
     * @return array
     */
    private function sanitize_list($value, int $maxitems, int $maxlength): array {
        if (!is_array($value)) {
            return [];
        }

        $result = [];
        foreach ($value as $item) {
            $text = $this->sanitize_text((string)$item, $maxlength);
            if ($text === '') {
                continue;
            }
            if (!in_array($text, $result, true)) {
                $result[] = $text;
            }
            if (count($result) >= $maxitems) {
                break;
            }
        }
        return $result;
    }

    /**
     * Sanitize text value and clamp by length.
     *
     * @param string $text
     * @param int $maxlength
     * @return string
     */
    private function sanitize_text(string $text, int $maxlength): string {
        $text = str_replace("\r\n", "\n", $text);
        $text = preg_replace('/\s+/', ' ', trim($text));
        if (!is_string($text) || $text === '') {
            return '';
        }
        return trim((string)\core_text::substr($text, 0, $maxlength));
    }

    /**
     * Clamp integer to [min, max].
     *
     * @param int $value
     * @param int $min
     * @param int $max
     * @return int
     */
    private function clamp_int(int $value, int $min, int $max): int {
        if ($value < $min) {
            return $min;
        }
        if ($value > $max) {
            return $max;
        }
        return $value;
    }

    /**
     * Clamp float to [min, max].
     *
     * @param float $value
     * @param float $min
     * @param float $max
     * @return float
     */
    private function clamp_float(float $value, float $min, float $max): float {
        if ($value < $min) {
            return $min;
        }
        if ($value > $max) {
            return $max;
        }
        return $value;
    }
}
