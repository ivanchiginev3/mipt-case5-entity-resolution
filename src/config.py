"""
Глобальные настройки проекта
"""

# мусорные записи
GARBAGE_PATTERNS = [
    'замените на',
    'согласно списку',
    'согласно по списку',
    'unknown',
    'неизвест',
]

# слишком короткие имена
MIN_NAME_LENGTH = 2

# допустимые скрипты
VALID_SCRIPTS = [
    'cyrillic',
    'latin',
    'armenian',
    'mixed'
]

# юридические формы
LEGAL_FORMS = {
    'ооо', 'зао', 'оао', 'пао',
    'llc', 'ltd', 'inc',
    'corp', 'company',
    'co', 'gmbh',
    'spy', 'pby'
}