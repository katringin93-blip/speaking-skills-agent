# ... (после того как собрали all_segments и нашли me_id) ...

                # 4. Сохранение и Анализ
                print("[4/4] Сохранение отчетов и запуск контекстного анализа...")
                full_text_lines = []
                
                for s in all_segments:
                    spk_label = "YOU" if s["global_id"] == me_id else s["global_id"]
                    line = f"[{s['start']:.1f}-{s['end']:.1f}] {spk_label}: {s.get('text','')}"
                    full_text_lines.append(line)
                
                # Записываем полный файл для анализа
                full_file = s_dir / "transcript_full.txt"
                full_file.write_text("\n".join(full_text_lines), encoding="utf-8")
                
                # Запускаем анализ (теперь передаем весь файл и ваш ID)
                print(">>> GPT анализирует вашу коммуникацию с бадди...")
                report = analyze_my_speech(api_key, full_file, me_id)
                
                # Сохраняем итоговый отчет
                report_file = s_dir / "ai_analysis_report.txt"
                report_file.write_text(report, encoding="utf-8")
                
                print("\n" + "="*40)
                print(f"АНАЛИЗ СЕССИИ ОТ {time.strftime('%Y-%m-%d')}")
                print("="*40)
                print(report)
                print("="*40)
